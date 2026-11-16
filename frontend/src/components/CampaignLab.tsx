import { BarChart3, Beaker, BookOpen, Database, Download, FileJson, Info, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  campaignFileUrl,
  downloadCampaignFile,
  fetchCampaign,
  startCampaign,
  type CampaignArm,
  type CampaignArtifact,
  type CampaignMetrics,
  type CampaignPayload
} from "../lib/api";
import type { SystemInfo } from "../types";
import { CampaignHelpDialog } from "./CampaignHelpDialog";
import { InfoTip } from "./InfoTip";

type CampaignLabProps = {
  systemInfo: SystemInfo | null;
};

type DatasetMode = "generated" | "manifest";

type TunerId =
  | "fixed_default"
  | "random_search"
  | "classic_ego"
  | "noisy_contextual_ego"
  | "smac3"
  | "smac3_ac"
  | "irace";

const TUNER_META: Record<TunerId, { label: string; help: string }> = {
  fixed_default: {
    label: "Configuração fixa padrão",
    help: "Baseline essencial: roda o SA com os parâmetros padrão, sem aprender nem testar configurações candidatas. Todo ganho das outras estratégias é medido contra ele."
  },
  random_search: {
    label: "Random Search",
    help: "Baseline de sanidade: sorteia configurações do SA no espaço de busca e guarda a melhor no treino."
  },
  classic_ego: {
    label: "EGO clássico",
    help: "Otimização bayesiana sem olhar as características da instância; serve para medir se o contexto realmente ajuda."
  },
  noisy_contextual_ego: {
    label: "EGO contextual noise-aware",
    help: "Proposta principal: usa características da instância, modela ruído entre seeds e escolhe configurações via GP/EGO."
  },
  smac3: {
    label: "SMAC3 (HPO)",
    help: "Configurador externo de referência no modo de otimização de hiperparâmetros (média sobre instâncias). Requer pip install '.[tuners]'; se não estiver instalado, aparece como pulado."
  },
  smac3_ac: {
    label: "SMAC3 (AC, com instâncias)",
    help: "SMAC3 no modo de configuração de algoritmos: recebe as instâncias e suas features e faz racing entre elas, com o mesmo orçamento de runs. É o uso padrão do SMAC para configurar algoritmos."
  },
  irace: {
    label: "irace",
    help: "Configurador externo em R. Requer R + irace no PATH; sem isso, a campanha é preparada para execução externa."
  }
};

const ARM_LABELS: Record<string, string> = {
  fixed_default: "Config. fixa padrão",
  random_search: "Random Search",
  classic_ego: "EGO clássico",
  noisy_contextual_ego: "EGO contextual",
  smac3: "SMAC3 HPO",
  smac3_ac: "SMAC3 AC",
  irace: "irace",
  per_class: "Por classe",
  nearest_neighbor: "kNN",
  ablation_min_y_incumbent: "Ablation: min(y)",
  ablation_gp_fixed_hyperparams: "Ablation: GP fixo"
};

function formatLoss(value: number | null) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function formatInteger(value: number) {
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 }).format(value);
}

function formatMetric(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function formatPercent(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${Math.round(value * 100)}%`;
}

function parseSeeds(text: string) {
  return [...new Set(text.split(/[\s,;]+/).map((part) => Number.parseInt(part, 10)).filter(Number.isFinite))];
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }
  if (seconds < 60) {
    return `${Math.ceil(seconds)}s`;
  }
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) {
    return `${minutes}min`;
  }
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest > 0 ? `${hours}h ${rest}min` : `${hours}h`;
}

function estimateSolverRuns(
  selectedTuners: TunerId[],
  ablations: boolean,
  perInstanceBaselines: boolean,
  classesLimit: number,
  perClassTrain: number,
  seedCount: number,
  trials: number
) {
  const trainRunsPerTrial = Math.max(0, classesLimit) * Math.max(0, perClassTrain) * seedCount;
  const evaluationRuns = Math.max(0, classesLimit) * 2 * seedCount;
  let runs = 0;
  if (selectedTuners.includes("fixed_default")) {
    runs += evaluationRuns;
  }
  runs += selectedTuners.filter((tuner) => tuner !== "fixed_default").length * (trainRunsPerTrial * trials + evaluationRuns);
  if (ablations) {
    runs += 2 * (trainRunsPerTrial * trials + evaluationRuns);
  }
  if (perInstanceBaselines) {
    runs += 2 * (trainRunsPerTrial * trials + evaluationRuns);
  }
  return runs;
}

function validationDelta(value: number | null, bestValidation: number | null) {
  if (value === null || bestValidation === null) {
    return "";
  }
  if (value === bestValidation) {
    return "melhor";
  }
  if (Math.abs(bestValidation) < 1e-12) {
    return `+${(value - bestValidation).toFixed(4)}`;
  }
  return `+${(((value - bestValidation) / Math.abs(bestValidation)) * 100).toFixed(1)}%`;
}

function isCompleteValidation(arm: CampaignArm) {
  const metrics = arm.validation_metrics;
  return (
    arm.status === "ok" &&
    arm.validation_loss !== null &&
    metrics !== null &&
    metrics !== undefined &&
    metrics.runs > 0 &&
    metrics.feasible_runs === metrics.runs &&
    metrics.timeout_runs === 0 &&
    metrics.error_runs === 0
  );
}

function CampaignComparisonChart({
  arms,
  bestValidation
}: {
  arms: CampaignArm[];
  bestValidation: number | null;
}) {
  const comparable = arms.filter((arm) => arm.status === "ok" && arm.validation_loss !== null);
  if (comparable.length === 0) {
    return null;
  }
  const maxLoss = Math.max(
    1e-9,
    ...comparable.flatMap((arm) => [arm.validation_loss ?? 0, arm.test_loss ?? 0]).filter((value) => Number.isFinite(value))
  );
  return (
    <div className="campaign-chart-panel" aria-label="Gráfico de comparação da campanha">
      <div className="campaign-chart-heading">
        <BarChart3 size={16} aria-hidden="true" />
        <div>
          <strong>Comparação principal</strong>
          <span>Perda menor é melhor; validação escolhe o braço, teste confirma uma única vez.</span>
        </div>
      </div>
      <div className="campaign-chart">
        {comparable.map((arm) => {
          const validation = arm.validation_loss ?? 0;
          const test = arm.test_loss ?? 0;
          return (
            <div className="campaign-chart-row" key={arm.arm}>
              <div className="campaign-chart-label">
                <strong>{ARM_LABELS[arm.arm] ?? arm.arm}</strong>
                <span>
                  {isCompleteValidation(arm)
                    ? bestValidation === null
                      ? "diagnóstico"
                      : validationDelta(arm.validation_loss, bestValidation)
                    : "validação incompleta"}
                </span>
              </div>
              <div className="campaign-metric-bars">
                <div className="metric-bar-line">
                  <span>val</span>
                  <i data-kind="validation" style={{ width: `${Math.max(3, (validation / maxLoss) * 100)}%` }} />
                  <b>{formatLoss(arm.validation_loss)}</b>
                </div>
                <div className="metric-bar-line">
                  <span>teste</span>
                  <i data-kind="test" style={{ width: `${Math.max(3, (test / maxLoss) * 100)}%` }} />
                  <b>{formatLoss(arm.test_loss)}</b>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CampaignDatasetSummary({ dataset }: { dataset: NonNullable<CampaignPayload["summary"]>["dataset"] }) {
  if (!dataset) {
    return null;
  }
  const splitSummary = dataset.summary ?? {};
  const classes = dataset.classes ?? [];
  const curation = dataset.curation ?? null;
  const classSlots = curation?.class_slots ?? classes.length;
  const replacedSlots = curation?.class_slots_replaced ?? curation?.class_replacements ?? 0;
  return (
    <div className="campaign-result-context">
      <div className="campaign-result-heading">
        <FileJson size={16} aria-hidden="true" />
        <div>
          <strong>Dataset usado</strong>
          <span>
            {dataset.source === "manifest" ? "manifest local informado na tela" : "gerado automaticamente para esta campanha"}
          </span>
        </div>
      </div>
      <div className="dataset-facts">
        <span>{classSlots || "?"} classes</span>
        <span>{splitSummary.train ?? "?"} treino</span>
        <span>{splitSummary.validation ?? "?"} validação</span>
        <span>{splitSummary.test ?? "?"} teste</span>
      </div>
      <p className="manifest-path">
        Manifest: <code>{dataset.manifest_path}</code>
      </p>
      {curation ? (
        <p className="campaign-mini-note">
          Curadoria: {curation.accepted}/{curation.instances} instâncias aceitas no preflight; {curation.fallback} fallback(s);{" "}
          {curation.discriminative ?? 0} discriminativa(s); {replacedSlots}/{classSlots || "?"} slot(s) de classe substituído(s);{" "}
          LB médio {formatMetric(curation.mean_preflight_lower_bound, 2)}.
        </p>
      ) : null}
    </div>
  );
}

function metricLine(metrics: CampaignMetrics | null | undefined) {
  if (!metrics) {
    return "—";
  }
  const feasible =
    metrics.feasible_runs !== undefined && metrics.runs !== undefined
      ? ` · ${metrics.feasible_runs}/${metrics.runs} viáveis`
      : "";
  return `${formatMetric(metrics.mean_bins_used, 2)} bins · gap ${formatMetric(metrics.mean_gap_to_lower_bound, 3)}${feasible}`;
}

function runtimeLine(validation: CampaignMetrics | null | undefined, test: CampaignMetrics | null | undefined) {
  const val = validation?.mean_runtime_seconds;
  const tst = test?.mean_runtime_seconds;
  if (val == null && tst == null) {
    return "—";
  }
  return `${formatMetric(val, 2)}s / ${formatMetric(tst, 2)}s`;
}

function CampaignQualityPanel({ arms }: { arms: CampaignArm[] }) {
  const rows = arms.filter((arm) => arm.status === "ok" && (arm.validation_metrics || arm.test_metrics));
  if (rows.length === 0) {
    return null;
  }
  return (
    <div className="campaign-quality-panel" aria-label="Qualidade das soluções por braço">
      <div className="campaign-result-heading">
        <BarChart3 size={16} aria-hidden="true" />
        <div>
          <strong>Qualidade da solução</strong>
          <span>Caixas e gap explicam a perda; solutions.jsonl traz placements e convergência por run.</span>
        </div>
      </div>
      <div className="campaign-quality-table">
        <div className="campaign-quality-row campaign-quality-head">
          <span>Braço</span>
          <span>Validação</span>
          <span>Teste</span>
          <span>Ótimos</span>
          <span>Tempo/run (val/teste)</span>
          <span>Timeouts</span>
        </div>
        {rows.map((arm) => {
          const validation = arm.validation_metrics;
          const test = arm.test_metrics;
          const timeouts = (validation?.timeout_runs ?? 0) + (test?.timeout_runs ?? 0);
          return (
            <div className="campaign-quality-row" key={arm.arm}>
              <span>{ARM_LABELS[arm.arm] ?? arm.arm}</span>
              <span>{metricLine(validation)}</span>
              <span>{metricLine(test)}</span>
              <span>
                val {formatPercent(validation?.optimal_rate)} · teste {formatPercent(test?.optimal_rate)}
              </span>
              <span>{runtimeLine(validation, test)}</span>
              <span>{timeouts}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CampaignLab({ systemInfo }: CampaignLabProps) {
  const [datasetMode, setDatasetMode] = useState<DatasetMode>("generated");
  const [manifestPath, setManifestPath] = useState("");
  const [tuners, setTuners] = useState<Record<TunerId, boolean>>({
    fixed_default: true,
    random_search: true,
    classic_ego: true,
    noisy_contextual_ego: true,
    smac3: false,
    smac3_ac: false,
    irace: false
  });
  const [ablations, setAblations] = useState(false);
  const [perInstanceBaselines, setPerInstanceBaselines] = useState(false);
  // Defaults com potência estatística: o orçamento real deve ser Aval./run
  // (o tempo é só folga de segurança — se o tempo médio encostar no limite,
  // toda configuração roda "o que der no relógio" e os braços empatam);
  // trials suficientes para discriminar tuners e classes para dar variedade
  // de contexto.
  const [trials, setTrials] = useState(12);
  const [seedsText, setSeedsText] = useState("11, 23");
  const [maxEvaluations, setMaxEvaluations] = useState(1200);
  const [budgetMode, setBudgetMode] = useState<"evaluations" | "time">("evaluations");
  const [timeLimit, setTimeLimit] = useState(12);
  const [classesLimit, setClassesLimit] = useState(3);
  const [perClassTrain, setPerClassTrain] = useState(1);
  const [includeMinDistance, setIncludeMinDistance] = useState(false);
  const [persist, setPersist] = useState(false);
  const [job, setJob] = useState<CampaignPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);
  const pollRef = useRef<number | null>(null);
  const pollFailuresRef = useRef(0);

  const isRunning = Boolean(job && ["queued", "running"].includes(job.status));
  const useManifest = datasetMode === "manifest";
  const parsedSeeds = parseSeeds(seedsText);
  const selectedTuners = (Object.keys(TUNER_META) as TunerId[]).filter((tuner) => tuners[tuner]);
  // Modo "por avaliações": o relógio sai do caminho — o teto de tempo é
  // calculado automaticamente com folga generosa (~60 ms/avaliação, mínimo
  // 15 s) e serve apenas como salvaguarda; quem governa o orçamento são as
  // avaliações, então configurações diferentes produzem trajetórias completas
  // e comparáveis. Modo "por tempo": comportamento clássico de teto de parede.
  const autoTimeLimit = Math.max(15, Math.ceil(maxEvaluations * 0.06));
  const effectiveTimeLimit = budgetMode === "evaluations" ? autoTimeLimit : timeLimit;
  const estimatedRuns = useManifest
    ? null
    : estimateSolverRuns(selectedTuners, ablations, perInstanceBaselines, classesLimit, perClassTrain, parsedSeeds.length, trials);
  // Estimativa prática: no modo avaliações o run típico custa bem menos que o
  // teto (≈12 ms/aval + overhead); no modo tempo, o run tende a usar o teto.
  const expectedRunSeconds = budgetMode === "evaluations" ? maxEvaluations * 0.012 + 0.8 : timeLimit + 0.5;
  // A campanha roda os runs do solver em paralelo (até workers.recommended).
  // Speedup efetivo conservador: barreiras por trial (EGO) e desbalanceamento
  // impedem o ganho linear; ~70% de eficiência é uma estimativa prudente.
  const parallelWorkers = Math.max(1, systemInfo?.workers?.recommended ?? 1);
  const effectiveSpeedup = Math.max(1, 1 + (parallelWorkers - 1) * 0.7);
  const estimatedWallLimit = estimatedRuns === null ? 0 : (estimatedRuns * expectedRunSeconds) / effectiveSpeedup;
  const selectedArmCount = selectedTuners.length + (ablations ? 2 : 0) + (perInstanceBaselines ? 2 : 0);
  const trainInstances = classesLimit * perClassTrain;
  const validationInstances = classesLimit;
  const testInstances = classesLimit;
  const totalGeneratedInstances = trainInstances + validationInstances + testInstances;
  const seedList = parsedSeeds.length > 0 ? parsedSeeds.join(", ") : "nenhuma seed válida";
  const validationSeedList = parsedSeeds.length > 0 ? parsedSeeds.map((seed) => seed + 10_000).join(", ") : "—";
  const testSeedList = parsedSeeds.length > 0 ? parsedSeeds.map((seed) => seed + 20_000).join(", ") : "—";
  const generatedOccupancyScale = 1.6;

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) {
      return;
    }
    let disposed = false;
    pollRef.current = window.setInterval(() => {
      void fetchCampaign(job.job_id)
        .then((payload) => {
          if (disposed) {
            return;
          }
          pollFailuresRef.current = 0;
          setReconnecting(0);
          setJob(payload);
          setError(null);
        })
        .catch((requestError) => {
          if (disposed) {
            return;
          }
          // Falhas transitórias (ex.: backend reiniciando) não derrubam o
          // acompanhamento: o status é durável em disco e o poll re-encontra
          // o job. Só viramos erro depois de muitas tentativas seguidas.
          pollFailuresRef.current += 1;
          setReconnecting(pollFailuresRef.current);
          if (pollFailuresRef.current >= 15) {
            setError(
              requestError instanceof Error
                ? `Conexão perdida com o backend: ${requestError.message}`
                : "Conexão perdida com o backend"
            );
          }
        });
    }, 1200);
    return () => {
      disposed = true;
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
    };
  }, [job?.job_id, job?.status]);

  const handleStart = async () => {
    setError(null);
    const seeds = parseSeeds(seedsText);
    const selected = selectedTuners;
    if (selected.length === 0 || seeds.length === 0) {
      setError("Selecione ao menos uma estratégia e informe seeds válidas.");
      return;
    }
    if (useManifest && manifestPath.trim().length === 0) {
      setError("Informe o caminho do manifest.json do dataset local.");
      return;
    }
    const dataset = useManifest
      ? { manifest: manifestPath.trim() }
      : {
          generate: {
            per_class: { train: perClassTrain, validation: 1, test: 1 },
            seed: 7,
            include_min_distance: includeMinDistance,
            classes_limit: classesLimit,
            occupancy_scale: generatedOccupancyScale,
            prefer_discriminative: true,
            replace_rejected_classes: true,
            max_replacement_classes: 10,
            target_min_lower_bound: 4
          }
        };
    const spec = {
      dataset,
      seeds,
      trials,
      budget: { max_evaluations: maxEvaluations, time_limit_seconds: effectiveTimeLimit, temperature_calibration_samples: 6 },
      tuners: selected,
      ablations,
      per_instance_baselines: perInstanceBaselines,
      persist
    };
    try {
      const payload = await startCampaign(spec);
      setJob(payload);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao iniciar campanha");
    }
  };

  const handleDownload = async (artifact: CampaignArtifact) => {
    if (!job) {
      return;
    }
    try {
      await downloadCampaignFile(job.job_id, artifact);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : `Falha ao baixar ${artifact}`);
    }
  };

  const summary = job?.summary ?? null;
  const arms: CampaignArm[] = summary?.arms ?? [];
  const validArms = arms.filter(isCompleteValidation);
  const datasetDiscriminationWarning = summary?.dataset_discrimination_warning ?? null;
  const bestValidation =
    validArms.length > 0 && !datasetDiscriminationWarning
      ? Math.min(...validArms.map((arm) => arm.validation_loss as number))
      : null;
  const okValidationArms = arms.filter((arm) => arm.status === "ok" && arm.validation_loss !== null);
  const hasIncompleteValidation = okValidationArms.some((arm) => !isCompleteValidation(arm));
  const inferredComparisonWarning = hasIncompleteValidation
    ? "A validação tem runs não viáveis ou em timeout. Use estes números só como diagnóstico; rode novamente antes de interpretar ranking."
    : null;
  const comparisonWarning = summary?.comparison_warning ?? inferredComparisonWarning;
  const comparisonValid =
    summary?.comparison_valid ?? (okValidationArms.length > 0 && !hasIncompleteValidation && !datasetDiscriminationWarning);

  return (
    <div className="experiments-page">
      <section className="panel experiment-setup" aria-label="Configuração da campanha">
        <div className="panel-heading split-heading">
          <div className="heading-label">
            <Beaker size={18} aria-hidden="true" />
            <h2>Campanha de estratégias</h2>
            <InfoTip text="Compara configuradores do Simulated Annealing. Por padrão a campanha NÃO usa a instância da aba Solver: ela cria um dataset estratificado com treino, validação e teste." />
          </div>
          <button className="ghost-button compact-button campaign-help-trigger" type="button" onClick={() => setHelpOpen(true)}>
            <BookOpen size={15} aria-hidden="true" />
            Ajuda
          </button>
        </div>

        <div className="campaign-section">
          <div className="campaign-section-title">
            <Database size={16} aria-hidden="true" />
            <div>
              <strong>Origem dos dados</strong>
              <span>{useManifest ? "usar dataset salvo no computador" : "gerar dataset sintético estratificado"}</span>
            </div>
          </div>
          <div className="segmented compact dataset-mode-switch" role="group" aria-label="Origem do dataset">
            <button type="button" className={!useManifest ? "active" : ""} onClick={() => setDatasetMode("generated")} disabled={isRunning}>
              Gerado
            </button>
            <button type="button" className={useManifest ? "active" : ""} onClick={() => setDatasetMode("manifest")} disabled={isRunning}>
              Manifest local
            </button>
          </div>
          {useManifest ? (
            <>
              <label className="manifest-field">
                <span>Manifest JSON</span>
                <input
                  type="text"
                  value={manifestPath}
                  placeholder="/Users/.../manifest.json"
                  disabled={isRunning}
                  onChange={(event) => setManifestPath(event.target.value)}
                />
              </label>
              <p className="campaign-mini-note">
                Use um manifest exportado por campanha anterior ou gerado pela CLI. Ele deve apontar para splits
                <code> train</code>, <code>validation</code> e <code>test</code>.
              </p>
            </>
          ) : (
            <>
              <div className="dataset-facts">
                <span>{classesLimit} classes</span>
                <span>{trainInstances} treino</span>
                <span>{validationInstances} validação</span>
                <span>{testInstances} teste</span>
              </div>
              <p className="campaign-mini-note">
                Dataset criado na pasta da campanha com curadoria discriminativa. A instância aberta na aba Solver serve
                para rodadas individuais, não para esta comparação.
              </p>
            </>
          )}
        </div>

        <div className="variant-grid">
          {(Object.keys(TUNER_META) as TunerId[]).map((tuner) => (
            <label className="switch-line" key={tuner}>
              <input
                type="checkbox"
                checked={tuners[tuner]}
                onChange={(event) => setTuners({ ...tuners, [tuner]: event.target.checked })}
                disabled={isRunning}
              />
              <span>
                {TUNER_META[tuner].label}
                <InfoTip text={TUNER_META[tuner].help} />
              </span>
            </label>
          ))}
          <label className="switch-line">
            <input type="checkbox" checked={ablations} onChange={(event) => setAblations(event.target.checked)} disabled={isRunning} />
            <span>
              Ablations
              <InfoTip text="Adiciona braços que desligam um componente por vez: incumbente min(y) em vez de plug-in, e GP sem otimização de hiperparâmetros. Ajuda a entender qual peça gerou o ganho." />
            </span>
          </label>
          <label className="switch-line">
            <input type="checkbox" checked={perInstanceBaselines} onChange={(event) => setPerInstanceBaselines(event.target.checked)} disabled={isRunning} />
            <span>
              Baselines por instância (classe / kNN)
              <InfoTip text="Adiciona dois concorrentes simples do EGO contextual: 'config. por classe' (melhor config média de treino na classe da instância) e 'vizinho mais próximo / kNN' (copia a melhor config da instância de treino mais parecida). São os baselines exigidos pelo parecer para saber se o GP contextual agrega valor além de um mapa classe→config." />
            </span>
          </label>
        </div>

        <div className="campaign-section">
          <div className="campaign-section-title">
            <BarChart3 size={16} aria-hidden="true" />
            <div>
              <strong>Orçamento por run</strong>
              <span>
                {budgetMode === "evaluations"
                  ? "avaliações mandam; o relógio é só salvaguarda automática (recomendado)"
                  : "teto de tempo manual — atenção à saturação"}
              </span>
            </div>
            <InfoTip text="Por avaliações (recomendado para comparar configuradores): cada run executa o mesmo número de avaliações do SA e o teto de tempo é calculado automaticamente com folga — o relógio nunca corta, então a diferença entre braços vem da configuração, não do tempo. Por tempo: o run para no teto de parede; se o tempo médio encostar no limite, todos os braços empatam." />
          </div>
          <div className="segmented compact dataset-mode-switch" role="group" aria-label="Modo de orçamento">
            <button
              type="button"
              className={budgetMode === "evaluations" ? "active" : ""}
              onClick={() => setBudgetMode("evaluations")}
              disabled={isRunning}
            >
              Por avaliações ✓
            </button>
            <button
              type="button"
              className={budgetMode === "time" ? "active" : ""}
              onClick={() => setBudgetMode("time")}
              disabled={isRunning}
            >
              Por tempo
            </button>
          </div>
        </div>

        <div className="field-grid">
          <label>
            <span className="label-with-info">
              Trials
              <InfoTip text="Número de receitas de parâmetros que cada estratégia tenta no treino. Ex.: 6 trials = 6 receitas testadas antes de escolher a melhor." />
            </span>
            <input type="number" min={2} value={trials} disabled={isRunning} onChange={(event) => setTrials(Math.max(2, Number(event.target.value)))} />
          </label>
          <label>
            <span className="label-with-info">
              Seeds
              <InfoTip text="Números que controlam os sorteios do solver. A mesma seed repete o mesmo caminho; várias seeds reduzem o risco de concluir algo por sorte." />
            </span>
            <input type="text" value={seedsText} disabled={isRunning} onChange={(event) => setSeedsText(event.target.value)} />
          </label>
          <label>
            <span className="label-with-info">
              Aval./run
              <InfoTip text="Máximo de soluções candidatas testadas pelo SA em uma execução: 1 instância × 1 seed × 1 configuração." />
            </span>
            <input type="number" min={50} value={maxEvaluations} disabled={isRunning} onChange={(event) => setMaxEvaluations(Math.max(50, Number(event.target.value)))} />
          </label>
          {budgetMode === "evaluations" ? (
            <label>
              <span className="label-with-info">
                Tempo/run (s)
                <InfoTip text="Calculado automaticamente com folga generosa (~60 ms por avaliação, mínimo 15 s). Serve só como salvaguarda contra travamentos — o orçamento real são as avaliações, então o relógio não interfere na comparação entre braços." />
              </span>
              <input type="text" value={`automático ~${autoTimeLimit}s`} disabled readOnly />
            </label>
          ) : (
            <label>
              <span className="label-with-info">
                Tempo/run (s)
                <InfoTip text="Teto manual de tempo por run. Cuidado: se o tempo médio/run encostar neste limite (veja a coluna Tempo/run nos resultados), toda configuração roda 'o que der no relógio' e os braços empatam — aumente este valor, reduza Aval./run ou volte ao modo 'Por avaliações'." />
              </span>
              <input type="number" min={1} value={timeLimit} disabled={isRunning} onChange={(event) => setTimeLimit(Math.max(1, Number(event.target.value)))} />
            </label>
          )}
          <label>
            <span className="label-with-info">
              Classes
              <InfoTip text="Quantidade de famílias sintéticas de instâncias. Cada classe combina tamanho, ocupação, forma, heterogeneidade e, opcionalmente, min_distance." />
            </span>
            <input type="number" min={1} max={72} value={classesLimit} disabled={isRunning || useManifest} onChange={(event) => setClassesLimit(Math.max(1, Math.min(72, Number(event.target.value))))} />
          </label>
          <label>
            <span className="label-with-info">
              Train/classe
              <InfoTip text="Exemplos de treino por tipo de problema. Ex.: 4 classes e Train/classe 2 geram 8 instâncias de treino." />
            </span>
            <input type="number" min={1} max={5} value={perClassTrain} disabled={isRunning || useManifest} onChange={(event) => setPerClassTrain(Math.max(1, Math.min(5, Number(event.target.value))))} />
          </label>
        </div>

        <label className="switch-line persist-line">
          <input type="checkbox" checked={includeMinDistance} onChange={(event) => setIncludeMinDistance(event.target.checked)} disabled={isRunning || useManifest} />
          <span>
            Incluir classes com min_distance
            <InfoTip text="Permite selecionar classes com restrição de distância mínima entre pares de itens. Só se aplica ao dataset gerado." />
          </span>
        </label>
        <label className="switch-line persist-line">
          <input type="checkbox" checked={persist} onChange={(event) => setPersist(event.target.checked)} disabled={isRunning} />
          <span>
            Persistir execuções (campaign_id no SQLite)
            <InfoTip text="Grava cada execução no banco com campaign_id/experiment_id para auditoria. Desligado por padrão para não enviesar o aprendizado online." />
          </span>
        </label>

        <div className="campaign-plan">
          <div className="campaign-flow" aria-label="Fluxo da campanha">
            <span>Dataset</span>
            <i />
            <span>Treino</span>
            <i />
            <span>Validação</span>
            <i />
            <span>Teste</span>
          </div>
          <dl>
            <div>
              <dt>Dados</dt>
              <dd>
                {useManifest
                  ? "contagens lidas do manifest ao iniciar"
                  : `${totalGeneratedInstances} instâncias: ${trainInstances} treino, ${validationInstances} validação, ${testInstances} teste`}
              </dd>
            </div>
            <div>
              <dt>Seeds</dt>
              <dd>
                treino {seedList}; validação {validationSeedList}; teste {testSeedList}
              </dd>
            </div>
            <div>
              <dt>Modelo</dt>
              <dd>o solver é SA+Extreme Points; EGO/SMAC/irace só escolhem parâmetros do SA</dd>
            </div>
          </dl>
        </div>

        {systemInfo ? (
          <div className="experiment-summary">
            Máquina: {systemInfo.os} {systemInfo.architecture}, {systemInfo.cpu_physical ?? "?"} núcleos — a campanha
            paraleliza os runs do solver em até {parallelWorkers} núcleo(s), deixando 1 livre para o sistema. O número
            de núcleos muda só o tempo de parede, não os resultados (cada run é determinístico por instância/config/seed).
            Acompanhe o consumo na aba Sistema.
          </div>
        ) : null}
        <div className="experiment-summary campaign-estimate">
          <span>{selectedArmCount} braços</span>
          <span>{estimatedRuns === null ? "runs calculados ao iniciar" : `${formatInteger(estimatedRuns)} runs do solver`}</span>
          <span>{estimatedRuns === null ? "tempo depende do manifest" : `estimativa prática: ${formatDuration(estimatedWallLimit)}`}</span>
        </div>

        {isRunning && job ? (
          <div className="run-progress" data-status="running">
            <div className="progress-heading">
              <span>
                {job.progress.stage}: {job.progress.detail}
              </span>
              <strong>{Math.round(job.progress.fraction * 100)}%</strong>
            </div>
            <div className="progress-track">
              <i style={{ width: `${Math.min(100, job.progress.fraction * 100)}%` }} />
            </div>
            {job.progress.estimated_solver_runs ? (
              <div className="progress-meta">
                <span>
                  {job.progress.completed_solver_runs ?? 0}/{job.progress.estimated_solver_runs} runs
                </span>
                <span>{job.progress.current_phase ?? job.progress.stage}</span>
              </div>
            ) : null}
          </div>
        ) : null}

        {reconnecting > 0 && !error ? (
          <p className="reconnect-line" role="status">
            Conexão instável com o backend — tentando reconectar ({reconnecting}ª tentativa)… o progresso da campanha
            é preservado em disco.
          </p>
        ) : null}
        {error ? <p className="error-line">{error}</p> : null}
        {job?.status === "error" ? <p className="error-line">{job.error}</p> : null}

        <div className="actions">
          <button className="primary-action" type="button" onClick={() => void handleStart()} disabled={isRunning}>
            <Play size={17} aria-hidden="true" />
            {isRunning ? "Campanha em execução…" : "Rodar campanha"}
          </button>
        </div>
      </section>

      {helpOpen ? (
        <CampaignHelpDialog
          onClose={() => setHelpOpen(false)}
          datasetMode={datasetMode}
          classesLimit={classesLimit}
          perClassTrain={perClassTrain}
          trainInstances={trainInstances}
          validationInstances={validationInstances}
          testInstances={testInstances}
          totalGeneratedInstances={totalGeneratedInstances}
          trials={trials}
          seedList={seedList}
          validationSeedList={validationSeedList}
          testSeedList={testSeedList}
          maxEvaluations={maxEvaluations}
          timeLimit={effectiveTimeLimit}
          occupancyScale={generatedOccupancyScale}
          includeMinDistance={includeMinDistance}
          persist={persist}
          selectedArmCount={selectedArmCount}
          estimatedRuns={estimatedRuns}
          estimatedDuration={formatDuration(estimatedWallLimit)}
        />
      ) : null}

      <section className="panel experiment-results" aria-label="Resultados da campanha">
        <div className="panel-heading">
          <Beaker size={18} aria-hidden="true" />
          <h2>Resultados</h2>
          <InfoTip text="A coluna comparável é Validação (e Teste, usado uma única vez): o incumbente de cada braço reavaliado nas mesmas seeds frescas. A perda de treino é viesada por seleção (winner's curse)." />
        </div>

        {summary ? (
          <>
            {comparisonValid === false && comparisonWarning ? (
              <p className="reconnect-line" role="alert">
                Atenção: {comparisonWarning}
              </p>
            ) : null}
            {datasetDiscriminationWarning ? (
              <p className="reconnect-line" role="alert">
                Atenção: {datasetDiscriminationWarning}
              </p>
            ) : null}
            {summary.time_budget_saturated && summary.time_budget_warning ? (
              <p className="reconnect-line" role="alert">
                Atenção: {summary.time_budget_warning}
              </p>
            ) : null}
            <CampaignDatasetSummary dataset={summary.dataset} />
            {job && summary.artifacts?.comparison_png && comparisonValid ? (
              <figure className="campaign-publication-figure">
                <img
                  src={campaignFileUrl(job.job_id, "comparison.png")}
                  alt="Gráfico de comparação dos configuradores em qualidade de publicação"
                />
                <figcaption>
                  Figura em qualidade de publicação (200 dpi) gerada pela campanha — baixe junto com a tabela LaTeX
                  para usar direto no artigo.
                </figcaption>
              </figure>
            ) : (
              <CampaignComparisonChart arms={arms} bestValidation={bestValidation} />
            )}
            <CampaignQualityPanel arms={arms} />
            <div className="exp-table" role="table" aria-label="Comparação de braços">
              <div className="exp-row exp-head exp-row-campaign" role="row">
                <span>Braço</span>
                <span>Validação</span>
                <span>Teste</span>
                <span>Treino*</span>
                <span>Tuning</span>
                <span>Status</span>
              </div>
              {arms.map((arm) => (
                <div className="exp-row exp-row-campaign" role="row" key={arm.arm}>
                  <span>
                    {ARM_LABELS[arm.arm] ?? arm.arm}
                    {isCompleteValidation(arm) && arm.validation_loss === bestValidation ? " ★" : ""}
                  </span>
                  <span>{formatLoss(arm.validation_loss)}</span>
                  <span>{formatLoss(arm.test_loss)}</span>
                  <span>{formatLoss(arm.training_best_loss)}</span>
                  <span>
                    {arm.status !== "ok"
                      ? "—"
                      : arm.arm === "fixed_default"
                        ? "baseline (sem tuning)"
                        : `${arm.tuning_seconds.toFixed(1)}s · ${arm.tuning_solver_runs} runs`}
                  </span>
                  <span>{arm.status === "ok" ? "ok" : arm.reason ?? arm.status}</span>
                </div>
              ))}
            </div>
            <div className="convergence-stats">
              <span>{summary.records} execuções registradas</span>
              {summary.estimated_solver_runs ? <span>{summary.estimated_solver_runs} runs estimados</span> : null}
              <span>{summary.elapsed_seconds.toFixed(1)}s no total</span>
              <span>id: {summary.campaign_id}</span>
            </div>
            <div className="toolbar-actions campaign-downloads">
              <button className="ghost-button" type="button" onClick={() => void handleDownload("runs.csv")}>
                <Download size={15} aria-hidden="true" /> runs.csv
              </button>
              <button className="ghost-button" type="button" onClick={() => void handleDownload("solutions.jsonl")}>
                <Download size={15} aria-hidden="true" /> solutions.jsonl
              </button>
              <button className="ghost-button" type="button" onClick={() => void handleDownload("report.md")}>
                <Download size={15} aria-hidden="true" /> report.md
              </button>
              <button className="ghost-button" type="button" onClick={() => void handleDownload("metadata.json")}>
                <Download size={15} aria-hidden="true" /> metadata.json
              </button>
              <button className="ghost-button" type="button" onClick={() => void handleDownload("dataset_manifest.json")}>
                <Download size={15} aria-hidden="true" /> manifest.json
              </button>
              {summary.artifacts?.comparison_png ? (
                <button className="ghost-button publication-button" type="button" onClick={() => void handleDownload("comparison.png")}>
                  <Download size={15} aria-hidden="true" /> comparison.png (artigo)
                </button>
              ) : null}
              {summary.artifacts?.results_table_tex ? (
                <button className="ghost-button publication-button" type="button" onClick={() => void handleDownload("results_table.tex")}>
                  <Download size={15} aria-hidden="true" /> results_table.tex (artigo)
                </button>
              ) : null}
            </div>
          </>
        ) : (
          <div className="analysis-empty">
            Configure as estratégias e rode a campanha. Piloto rápido: dataset gerado, 4 estratégias padrão, 1 classe,
            2 trials e seed “11”. A instância da aba Solver não é usada aqui.
          </div>
        )}

        <div className="analysis-note">
          <Info size={15} aria-hidden="true" />
          <span>
            Piloto antes da campanha grande: confirme se EGO contextual supera Random Search em validação. O{" "}
            <strong>teste</strong> deve ser interpretado uma única vez; repetir olhando o teste invalida o protocolo.
          </span>
        </div>
      </section>
    </div>
  );
}
