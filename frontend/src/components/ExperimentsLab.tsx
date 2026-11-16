import { FlaskConical, Info, Play, Square } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { cancelSolveJob, fetchSolveJob, startSolveJob } from "../lib/api";
import type { ProblemInstance, SolveJobPayload, SolverConfig, SolverResult, SystemInfo } from "../types";
import { CampaignLab } from "./CampaignLab";
import { InfoTip } from "./InfoTip";

type ExperimentsLabProps = {
  instance: ProblemInstance | null;
  baseConfig: SolverConfig;
  baseSeed: number;
  systemInfo: SystemInfo | null;
};

type VariantId = "sequential" | "process_pool" | "cooperative";

type VariantResult = {
  variant: VariantId;
  seed: number;
  result: SolverResult;
};

type RunningState = {
  variantLabel: string;
  seed: number;
  completed: number;
  total: number;
  percent: number;
} | null;

const VARIANT_META: Record<VariantId, { label: string; help: string }> = {
  sequential: {
    label: "Sequencial",
    help: "Uma trajetória por vez, em um único processo. É a referência (baseline) para medir o speedup dos modos paralelos."
  },
  process_pool: {
    label: "Pool independente",
    help: "Multi-start dividido entre processos isolados, sem comunicação. Reprodutível seed a seed."
  },
  cooperative: {
    label: "Pool cooperativo",
    help: "Workers trocam a melhor solução global via memória central. Pode convergir mais rápido, mas é não-determinístico: compare por média de várias seeds."
  }
};

function formatNumber(value: number, digits = 2) {
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: digits }).format(value);
}

function parseSeeds(text: string): number[] {
  return [...new Set(
    text
      .split(/[\s,;]+/)
      .map((part) => Number.parseInt(part, 10))
      .filter((value) => Number.isFinite(value))
  )];
}

function variantConfig(variant: VariantId, base: SolverConfig, workers: number, syncInterval: number): SolverConfig {
  if (variant === "sequential") {
    return { ...base, parallel_mode: "sequential", parallel_workers: 1 };
  }
  if (variant === "process_pool") {
    return { ...base, parallel_mode: "process_pool", parallel_workers: workers };
  }
  return {
    ...base,
    parallel_mode: "cooperative_process_pool",
    parallel_workers: workers,
    cooperative_sync_interval: syncInterval
  };
}

async function waitForJob(jobId: string, shouldStop: () => boolean, onProgress: (percent: number) => void): Promise<SolveJobPayload> {
  for (;;) {
    if (shouldStop()) {
      try {
        return await cancelSolveJob(jobId);
      } catch {
        // ignore: o job pode já ter terminado
      }
    }
    const payload = await fetchSolveJob(jobId);
    onProgress(payload.progress.percent);
    if (["completed", "cancelled", "error"].includes(payload.status)) {
      return payload;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 600));
  }
}

export function ExperimentsLab({ instance, baseConfig, baseSeed, systemInfo }: ExperimentsLabProps) {
  const [labMode, setLabMode] = useState<"parallel" | "campaign">("parallel");
  const recommended = systemInfo?.workers.recommended ?? 2;
  const maximum = systemInfo?.workers.maximum ?? 8;

  const [enabled, setEnabled] = useState<Record<VariantId, boolean>>({
    sequential: true,
    process_pool: true,
    cooperative: true
  });
  const [workers, setWorkers] = useState(Math.max(2, recommended));
  const [syncInterval, setSyncInterval] = useState(25);
  const [seedsText, setSeedsText] = useState(`${baseSeed}, ${baseSeed + 1}, ${baseSeed + 2}`);
  const [maxEvaluations, setMaxEvaluations] = useState(baseConfig.max_evaluations);
  const [timeLimit, setTimeLimit] = useState(baseConfig.time_limit_seconds);
  const [persist, setPersist] = useState(false);
  const [running, setRunning] = useState<RunningState>(null);
  const [results, setResults] = useState<VariantResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef(false);

  const seeds = useMemo(() => parseSeeds(seedsText), [seedsText]);
  const variants = (Object.keys(VARIANT_META) as VariantId[]).filter((variant) => enabled[variant]);
  const totalRuns = variants.length * seeds.length;

  const aggregates = useMemo(() => {
    const byVariant = new Map<VariantId, VariantResult[]>();
    for (const row of results) {
      const bucket = byVariant.get(row.variant) ?? [];
      bucket.push(row);
      byVariant.set(row.variant, bucket);
    }
    const aggregateList = [...byVariant.entries()].map(([variant, rows]) => {
      const bins = rows.map((row) => row.result.best_found_bins);
      const runtimes = rows.map((row) => row.result.runtime_seconds);
      const evaluations = rows.map((row) => row.result.evaluations);
      const meanRuntime = runtimes.reduce((a, b) => a + b, 0) / rows.length;
      const meanEvaluations = evaluations.reduce((a, b) => a + b, 0) / rows.length;
      return {
        variant,
        runs: rows.length,
        meanBins: bins.reduce((a, b) => a + b, 0) / rows.length,
        bestBins: Math.min(...bins),
        meanGap: rows.reduce((a, row) => a + row.result.gap_to_lower_bound, 0) / rows.length,
        meanRuntime,
        evaluationsPerSecond: meanRuntime > 0 ? meanEvaluations / meanRuntime : 0
      };
    });
    const sequentialAggregate = aggregateList.find((item) => item.variant === "sequential");
    return aggregateList.map((item) => ({
      ...item,
      speedup:
        sequentialAggregate && item.meanRuntime > 0 && item.variant !== "sequential"
          ? sequentialAggregate.meanRuntime / item.meanRuntime
          : null
    }));
  }, [results]);

  const maxMeanRuntime = Math.max(...aggregates.map((item) => item.meanRuntime), 1e-9);

  const runExperiment = async () => {
    if (!instance) {
      setError("Carregue uma instância válida na aba Solver antes de comparar.");
      return;
    }
    if (variants.length === 0 || seeds.length === 0) {
      setError("Selecione pelo menos uma variante e informe seeds válidas (ex.: 1, 2, 3).");
      return;
    }
    setError(null);
    setResults([]);
    stopRef.current = false;
    let completed = 0;
    try {
      for (const variant of variants) {
        for (const seed of seeds) {
          if (stopRef.current) {
            return;
          }
          const config = {
            ...variantConfig(variant, baseConfig, workers, syncInterval),
            max_evaluations: maxEvaluations,
            time_limit_seconds: timeLimit
          };
          setRunning({
            variantLabel: VARIANT_META[variant].label,
            seed,
            completed,
            total: totalRuns,
            percent: 0
          });
          const job = await startSolveJob(instance, config, seed, persist);
          const finished = await waitForJob(
            job.job_id,
            () => stopRef.current,
            (percent) =>
              setRunning((previous) => (previous ? { ...previous, percent } : previous))
          );
          if (finished.status === "completed" && finished.result) {
            const finalResult = finished.result;
            setResults((previous) => [...previous, { variant, seed, result: finalResult }]);
          } else if (finished.status === "error") {
            setError(`Variante ${VARIANT_META[variant].label}, seed ${seed}: ${finished.error ?? "erro"}`);
          }
          completed += 1;
        }
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao executar experimento");
    } finally {
      setRunning(null);
    }
  };

  const stopExperiment = () => {
    stopRef.current = true;
  };

  const modeSwitch = (
    <div className="lab-mode-bar">
      <div className="segmented compact lab-mode" role="group" aria-label="Tipo de experimento">
        <button type="button" className={labMode === "parallel" ? "active" : ""} onClick={() => setLabMode("parallel")}>
          Paralelismo
        </button>
        <button type="button" className={labMode === "campaign" ? "active" : ""} onClick={() => setLabMode("campaign")}>
          Campanha (estratégias)
        </button>
      </div>
      <InfoTip
        text={
          labMode === "parallel"
            ? "Compara modos de execução (sequencial × pool × cooperativo) na instância atual."
            : "Compara estratégias que escolhem parâmetros do SA (fixo, Random Search, EGOs, SMAC3, irace e ablations) em dataset estratificado com treino/validação/teste."
        }
      />
    </div>
  );

  if (labMode === "campaign") {
    return (
      <div className="lab-shell">
        {modeSwitch}
        <CampaignLab systemInfo={systemInfo} />
      </div>
    );
  }

  return (
    <div className="lab-shell">
      {modeSwitch}
    <div className="experiments-page">
      <section className="panel experiment-setup" aria-label="Configuração do experimento">
        <div className="panel-heading">
          <FlaskConical size={18} aria-hidden="true" />
          <h2>Experimento comparativo</h2>
          <InfoTip text="Roda a mesma instância com as variantes selecionadas e as mesmas seeds, em sequência, e compara qualidade e tempo. Ideal para medir o efeito do paralelismo nesta máquina." />
        </div>

        <div className="variant-grid">
          {(Object.keys(VARIANT_META) as VariantId[]).map((variant) => (
            <label className="switch-line" key={variant}>
              <input
                type="checkbox"
                checked={enabled[variant]}
                onChange={(event) => setEnabled({ ...enabled, [variant]: event.target.checked })}
                disabled={Boolean(running)}
              />
              <span>
                {VARIANT_META[variant].label}
                <InfoTip text={VARIANT_META[variant].help} />
              </span>
            </label>
          ))}
        </div>

        <div className="field-grid">
          <label>
            <span className="label-with-info">
              Workers
              <InfoTip
                text={`Processos paralelos para Pool/Cooperativo. Nesta máquina: recomendado ${recommended}, máximo ${maximum} (núcleos físicos).`}
              />
            </span>
            <input
              type="number"
              min={2}
              max={maximum}
              value={workers}
              disabled={Boolean(running)}
              onChange={(event) => setWorkers(Math.max(2, Math.min(maximum, Number(event.target.value))))}
            />
          </label>
          <label>
            <span className="label-with-info">
              Sync coop.
              <InfoTip text="Intervalo (em avaliações) entre trocas da melhor solução no modo cooperativo. Menor = mais comunicação." />
            </span>
            <input
              type="number"
              min={5}
              value={syncInterval}
              disabled={Boolean(running)}
              onChange={(event) => setSyncInterval(Math.max(5, Number(event.target.value)))}
            />
          </label>
          <label>
            <span className="label-with-info">
              Avaliações
              <InfoTip text="Orçamento de avaliações por execução, igual para todas as variantes (comparação justa)." />
            </span>
            <input
              type="number"
              min={100}
              value={maxEvaluations}
              disabled={Boolean(running)}
              onChange={(event) => setMaxEvaluations(Math.max(100, Number(event.target.value)))}
            />
          </label>
          <label>
            <span className="label-with-info">
              Tempo (s)
              <InfoTip text="Limite de tempo por execução, igual para todas as variantes." />
            </span>
            <input
              type="number"
              min={1}
              step={1}
              value={timeLimit}
              disabled={Boolean(running)}
              onChange={(event) => setTimeLimit(Math.max(1, Number(event.target.value)))}
            />
          </label>
        </div>

        <label className="seeds-field">
          <span className="label-with-info">
            Seeds
            <InfoTip text="Lista de seeds separadas por vírgula. Cada variante roda com todas as seeds; o modo cooperativo, por ser não-determinístico, deve ser avaliado pela média." />
          </span>
          <input
            type="text"
            value={seedsText}
            disabled={Boolean(running)}
            onChange={(event) => setSeedsText(event.target.value)}
            placeholder="ex.: 1, 2, 3"
          />
        </label>

        <label className="switch-line persist-line">
          <input
            type="checkbox"
            checked={persist}
            onChange={(event) => setPersist(event.target.checked)}
            disabled={Boolean(running)}
          />
          <span>
            Persistir execuções no SQLite
            <InfoTip text="Desligado por padrão: experimentos repetidos da mesma instância enviesariam o aprendizado online do GPR." />
          </span>
        </label>

        <div className="experiment-summary">
          {instance ? (
            <span>
              Instância <strong>{instance.instance_id}</strong> · {instance.items.length} itens · {totalRuns} execuções
              previstas
            </span>
          ) : (
            <span>Nenhuma instância válida carregada — ajuste na aba Solver.</span>
          )}
        </div>

        {running ? (
          <div className="run-progress" data-status="running">
            <div className="progress-heading">
              <span>
                {running.variantLabel} · seed {running.seed} ({running.completed + 1}/{running.total})
              </span>
              <strong>{Math.round(running.percent)}%</strong>
            </div>
            <div className="progress-track">
              <i style={{ width: `${Math.min(100, running.percent)}%` }} />
            </div>
          </div>
        ) : null}

        {error ? <p className="error-line">{error}</p> : null}

        <div className="actions">
          {running ? (
            <button className="primary-action danger-action" type="button" onClick={stopExperiment}>
              <Square size={17} aria-hidden="true" />
              Parar experimento
            </button>
          ) : (
            <button className="primary-action" type="button" onClick={() => void runExperiment()} disabled={!instance}>
              <Play size={17} aria-hidden="true" />
              Rodar comparação
            </button>
          )}
        </div>
      </section>

      <section className="panel experiment-results" aria-label="Resultados do experimento">
        <div className="panel-heading">
          <FlaskConical size={18} aria-hidden="true" />
          <h2>Resultados</h2>
          <InfoTip text="Médias sobre as seeds executadas. Speedup = tempo médio do Sequencial dividido pelo tempo médio da variante." />
        </div>

        {aggregates.length > 0 ? (
          <>
            <div className="exp-table" role="table" aria-label="Comparação de variantes">
              <div className="exp-row exp-head" role="row">
                <span>Variante</span>
                <span>Caixas (média)</span>
                <span>Melhor</span>
                <span>Gap médio</span>
                <span>Tempo médio</span>
                <span>Aval./s</span>
                <span>Speedup</span>
              </div>
              {aggregates.map((aggregate) => (
                <div className="exp-row" role="row" key={aggregate.variant}>
                  <span>{VARIANT_META[aggregate.variant].label}</span>
                  <span>{formatNumber(aggregate.meanBins, 2)}</span>
                  <span>{aggregate.bestBins}</span>
                  <span>{formatNumber(aggregate.meanGap * 100, 1)}%</span>
                  <span>{formatNumber(aggregate.meanRuntime, 2)}s</span>
                  <span>{formatNumber(aggregate.evaluationsPerSecond, 0)}</span>
                  <span>{aggregate.speedup ? `${formatNumber(aggregate.speedup, 2)}×` : "—"}</span>
                </div>
              ))}
            </div>

            <div className="exp-bars" aria-label="Tempo médio por variante">
              {aggregates.map((aggregate) => (
                <div className="exp-bar-row" key={aggregate.variant}>
                  <span>{VARIANT_META[aggregate.variant].label}</span>
                  <div className="exp-bar-track">
                    <i style={{ width: `${(aggregate.meanRuntime / maxMeanRuntime) * 100}%` }} />
                  </div>
                  <strong>{formatNumber(aggregate.meanRuntime, 2)}s</strong>
                </div>
              ))}
            </div>

            <div className="exp-run-list">
              {results.map((row, index) => (
                <div className="exp-run" key={`${row.variant}-${row.seed}-${index}`}>
                  <span>
                    {VARIANT_META[row.variant].label} · seed {row.seed}
                  </span>
                  <span>
                    {row.result.best_found_bins} caixa(s) · {formatNumber(row.result.runtime_seconds, 2)}s ·{" "}
                    {row.result.evaluations.toLocaleString("pt-BR")} aval.
                  </span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="analysis-empty">
            Configure as variantes e clique em “Rodar comparação”. Os resultados aparecem aqui com médias, melhor caso e
            speedup.
          </div>
        )}

        <div className="analysis-note">
          <Info size={15} aria-hidden="true" />
          <span>
            O modo cooperativo é não-determinístico (a ordem das trocas depende do escalonador do SO). Compare-o sempre
            pela <strong>média de várias seeds</strong>, nunca por uma execução isolada.
          </span>
        </div>
      </section>
    </div>
    </div>
  );
}
