import { Cpu, LoaderCircle, Network, Pause, Play, RotateCcw, Save, Share2, SlidersHorizontal, Sparkles, Square } from "lucide-react";
import { InfoTip } from "./InfoTip";
import type { SolveJobPayload, SolverConfig, SystemInfo } from "../types";

type SolverControlsProps = {
  config: SolverConfig;
  seed: number;
  persist: boolean;
  onConfigChange: (config: SolverConfig) => void;
  onSeedChange: (seed: number) => void;
  onPersistChange: (persist: boolean) => void;
  onSolve: () => void;
  onCancel: () => void;
  onReset: () => void;
  solveJob: SolveJobPayload | null;
  isJobActive: boolean;
  systemInfo: SystemInfo | null;
  onSuggest?: () => void;
};

export function SolverControls({
  config,
  seed,
  persist,
  onConfigChange,
  onSeedChange,
  onPersistChange,
  onSolve,
  onCancel,
  onReset,
  solveJob,
  isJobActive,
  systemInfo,
  onSuggest
}: SolverControlsProps) {
  const recommendedWorkers = systemInfo?.workers.recommended ?? 2;
  const maxWorkers = systemInfo?.workers.maximum ?? 8;
  const update = <K extends keyof SolverConfig>(key: K, value: SolverConfig[K]) => {
    onConfigChange({ ...config, [key]: value });
  };
  const setCore = (useNativeCore: boolean) => {
    onConfigChange({ ...config, use_native_core: useNativeCore });
  };
  const setParallelMode = (mode: SolverConfig["parallel_mode"]) => {
    onConfigChange({
      ...config,
      parallel_mode: mode,
      parallel_workers: mode === "sequential" ? 1 : Math.max(2, Math.min(maxWorkers, recommendedWorkers))
    });
  };
  const isParallel = config.parallel_mode !== "sequential";
  const isCooperative = config.parallel_mode === "cooperative_process_pool";
  const pauseSupported = isCooperative
    ? true
    : config.use_native_core
      ? systemInfo?.pause_signals_available ?? true
      : config.parallel_mode === "sequential";
  const impactText = isCooperative
    ? "Cooperativo: workers trocam a melhor solução via memória central. Converge mais rápido, mas é não-determinístico — compare por média de seeds."
    : config.use_native_core
      ? config.parallel_mode === "process_pool" && config.parallel_workers > 1
        ? "C++ em subprocessos paralelos: menor tempo de parede, maior uso de CPU/RAM."
        : "C++ standalone: menor overhead no caminho quente, com histórico amostrado da trajetória."
      : config.parallel_mode === "process_pool" && config.parallel_workers > 1
        ? "Python ProcessPool: multi-start isolado por processo, reprodutível seed a seed."
        : "Python sequencial: melhor diagnóstico e menor consumo de recursos.";
  const pauseHint = pauseSupported
    ? "Pausa: suportada neste modo."
    : "Pausa: indisponível neste modo (pool independente não tem canal de comunicação); Interromper descarta o resultado.";
  const machineHint = systemInfo
    ? `Esta máquina: ${systemInfo.cpu_physical ?? "?"} núcleos físicos (${systemInfo.architecture}, ${systemInfo.os}) — recomendado ${recommendedWorkers}, máx. ${maxWorkers}.`
    : null;
  const progress = solveJob?.progress;
  const progressPercent = Math.min(100, Math.max(0, progress?.percent ?? 0));
  const primaryLabel = isJobActive
    ? solveJob?.status === "paused"
      ? "Retomar"
      : solveJob?.pause_supported
        ? "Pausar"
        : "Interromper"
    : "Executar";
  const PrimaryIcon = isJobActive ? (solveJob?.status === "paused" ? Play : solveJob?.pause_supported ? Pause : Square) : Play;
  const statusText = solveJob
    ? solveJob.status === "paused"
      ? "Pausado"
      : solveJob.status === "queued"
        ? "Na fila"
        : solveJob.status === "running"
          ? "Executando"
          : solveJob.status
    : "";

  const label = (text: string, help: string) => (
    <span className="label-with-info">
      {text}
      <InfoTip text={help} />
    </span>
  );

  return (
    <section className="panel solver-panel" aria-label="Configuração do solver">
      <div className="panel-heading">
        <SlidersHorizontal size={18} aria-hidden="true" />
        <h2>Solver</h2>
        <InfoTip text="Configure orçamento, aleatoriedade, core de execução e paralelismo antes de rodar o solver." />
      </div>

      <div className="field-grid">
        <label>
          {label("Seed", "Controla a aleatoriedade. Mesma seed e mesma configuração tendem a reproduzir a mesma trajetória.")}
          <input
            type="number"
            value={seed}
            onChange={(event) => onSeedChange(Number(event.target.value))}
          />
        </label>
        <label>
          {label("Avaliações", "Número máximo de soluções candidatas avaliadas pelo SA. Mais avaliações podem melhorar a solução, mas aumentam o tempo.")}
          <input
            min={1}
            type="number"
            value={config.max_evaluations}
            onChange={(event) => update("max_evaluations", Number(event.target.value))}
          />
        </label>
        <label>
          {label("Tempo", "Limite de tempo em segundos. A execução para quando atingir tempo, avaliações ou critério de parada.")}
          <input
            min={0.1}
            step={0.1}
            type="number"
            value={config.time_limit_seconds}
            onChange={(event) => update("time_limit_seconds", Number(event.target.value))}
          />
        </label>
        <label>
          {label("Resfriamento", "Taxa que reduz a temperatura do SA. Valores maiores esfriam devagar e exploram mais.")}
          <input
            min={0.5}
            max={0.999}
            step={0.001}
            type="number"
            value={config.cooling_rate}
            onChange={(event) => update("cooling_rate", Number(event.target.value))}
          />
        </label>
        <label>
          {label("Iter./T", "Quantidade de movimentos testados antes de reduzir a temperatura.")}
          <input
            min={1}
            type="number"
            value={config.iterations_per_temperature}
            onChange={(event) => update("iterations_per_temperature", Number(event.target.value))}
          />
        </label>
        <label>
          {label(
            "Multi-start (buscas)",
            "Número de buscas independentes: o solver dispara N partidas, cada uma de uma solução inicial diferente, e fica com a melhor. Nos modos paralelos, as partidas são distribuídas entre os workers."
          )}
          <input
            min={1}
            type="number"
            value={config.multi_start_count}
            onChange={(event) => update("multi_start_count", Number(event.target.value))}
          />
        </label>
        <label>
          {label(
            "Workers",
            `Processos paralelos quando Pool/Coop está ativo. ${machineHint ?? "O solver é CPU-bound: use até o número de núcleos físicos."}`
          )}
          <input
            disabled={!isParallel}
            min={1}
            max={maxWorkers}
            type="number"
            value={config.parallel_workers}
            onChange={(event) => update("parallel_workers", Math.max(1, Math.min(maxWorkers, Number(event.target.value))))}
          />
        </label>
        <label>
          {label("Amostra hist.", "Intervalo de registro do histórico. 1 registra mais pontos; valores maiores deixam a resposta mais leve.")}
          <input
            min={1}
            type="number"
            value={config.convergence_sample_rate}
            onChange={(event) => update("convergence_sample_rate", Number(event.target.value))}
          />
        </label>
      </div>

      <div className="mode-block">
        <span className="mode-label">
          Core
          <InfoTip text="Python é mais fácil de auditar; C++ é o acelerador nativo para rodadas maiores." />
        </span>
        <div className="segmented compact" role="group" aria-label="Core do solver">
          <button
            type="button"
            className={!config.use_native_core ? "active" : ""}
            onClick={() => setCore(false)}
            aria-pressed={!config.use_native_core}
            title="Core Python"
          >
            Python
          </button>
          <button
            type="button"
            className={config.use_native_core ? "active" : ""}
            onClick={() => setCore(true)}
            aria-pressed={config.use_native_core}
            title="Core C++"
          >
            <Cpu size={15} aria-hidden="true" />
            C++
          </button>
        </div>
      </div>

      <div className="mode-block">
        <span className="mode-label">
          Paralelismo
          <InfoTip text="Sequencial: uma busca por vez (reprodutível, pausável). Pool: buscas independentes em processos isolados. Coop: workers compartilham a melhor solução via memória central (mais rápido para convergir, não-determinístico)." />
        </span>
        <div className="segmented compact" role="group" aria-label="Modo de paralelismo">
          <button
            type="button"
            className={config.parallel_mode === "sequential" ? "active" : ""}
            onClick={() => setParallelMode("sequential")}
            aria-pressed={config.parallel_mode === "sequential"}
            title="Execução sequencial"
          >
            Seq.
          </button>
          <button
            type="button"
            className={config.parallel_mode === "process_pool" ? "active" : ""}
            onClick={() => setParallelMode("process_pool")}
            aria-pressed={config.parallel_mode === "process_pool"}
            title="Pool de processos independentes"
          >
            <Network size={15} aria-hidden="true" />
            Pool
          </button>
          <button
            type="button"
            className={isCooperative ? "active" : ""}
            onClick={() => setParallelMode("cooperative_process_pool")}
            aria-pressed={isCooperative}
            title="Pool cooperativo com memória central"
          >
            <Share2 size={15} aria-hidden="true" />
            Coop
          </button>
        </div>
      </div>

      {isCooperative ? (
        <div className="mode-block">
          <span className="mode-label">
            Sync
            <InfoTip text="Intervalo (em avaliações) entre trocas da melhor solução entre os workers. Menor = mais comunicação e convergência conjunta; maior = mais diversidade." />
          </span>
          <input
            type="number"
            min={5}
            value={config.cooperative_sync_interval ?? 25}
            onChange={(event) => update("cooperative_sync_interval", Math.max(5, Number(event.target.value)))}
          />
        </div>
      ) : null}

      <div className="impact-box" aria-live="polite">
        {impactText}
        <small className="pause-hint" data-supported={pauseSupported ? "true" : "false"}>
          {pauseHint}
        </small>
        {isParallel && machineHint ? <small className="machine-hint">{machineHint}</small> : null}
      </div>

      {isJobActive && progress ? (
        <div className="run-progress" data-status={solveJob?.status ?? "running"}>
          <div className="progress-heading">
            <span>
              <LoaderCircle size={15} aria-hidden="true" className={solveJob?.status === "paused" ? "" : "spin"} />
              {statusText}
            </span>
            <strong>{Math.round(progressPercent)}%</strong>
          </div>
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent}>
            <i style={{ width: `${progressPercent}%` }} />
          </div>
          <div className="progress-meta">
            <span>
              Aval. {progress.evaluation.toLocaleString("pt-BR")} / {progress.max_evaluations.toLocaleString("pt-BR")}
            </span>
            <span>{progress.elapsed_seconds.toLocaleString("pt-BR", { maximumFractionDigits: 1 })}s</span>
          </div>
        </div>
      ) : null}

      <div className="toggle-row">
        <label className="switch-line">
          <input
            type="checkbox"
            checked={persist}
            onChange={(event) => onPersistChange(event.target.checked)}
          />
          <span>
            <Save size={16} aria-hidden="true" /> SQLite
            <InfoTip text="Quando ativo, salva a execução no banco e alimenta o aprendizado online do GPR." />
          </span>
        </label>
      </div>

      {onSuggest ? (
        <button className="ghost-button suggest-button" type="button" onClick={onSuggest} disabled={isJobActive}>
          <Sparkles size={15} aria-hidden="true" />
          <span>Sugerir configuração (EGO)</span>
          <InfoTip text="Fecha o ciclo do aprendizado: o GPR treinado com as execuções persistidas avalia a instância atual (contexto) e o Expected Improvement escolhe a configuração mais promissora. Com poucas observações, a sugestão é aleatória (aquecimento)." />
        </button>
      ) : null}

      <div className="actions">
        <button className="primary-action" type="button" onClick={onSolve}>
          <PrimaryIcon size={18} aria-hidden="true" />
          {primaryLabel}
        </button>
        {isJobActive ? (
          <button className="icon-action danger-action" type="button" onClick={onCancel} title="Interromper execução">
            <Square size={17} aria-hidden="true" />
          </button>
        ) : (
          <button className="icon-action" type="button" onClick={onReset} title="Restaurar exemplo">
            <RotateCcw size={18} aria-hidden="true" />
          </button>
        )}
      </div>
    </section>
  );
}
