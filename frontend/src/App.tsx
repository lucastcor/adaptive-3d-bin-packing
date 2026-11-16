import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { Activity, BookOpen, Boxes, FlaskConical, MonitorCog } from "lucide-react";
import { FloatingMonitor } from "./components/FloatingMonitor";
import { AnalysisWorkbench } from "./components/AnalysisWorkbench";
import { ExperimentsLab } from "./components/ExperimentsLab";
import { GuidePanel } from "./components/GuidePanel";
import { JsonEditor } from "./components/JsonEditor";
import { LearningPanel } from "./components/LearningPanel";
import { MetricsRail } from "./components/MetricsRail";
import { RunsPanel } from "./components/RunsPanel";
import { SolverControls } from "./components/SolverControls";
import { SystemPanel } from "./components/SystemPanel";
import {
  downloadSpreadsheetTemplate,
  exportInstanceSpreadsheet,
  exportResultSpreadsheet,
  cancelSolveJob,
  fetchDefaultConfig,
  fetchDefaultInstance,
  fetchLearningStatus,
  fetchRunBundle,
  fetchRuns,
  fetchSolveJob,
  fetchSystemInfo,
  importSpreadsheet,
  pauseSolveJob,
  resumeSolveJob,
  startSolveJob,
  suggestConfig
} from "./lib/api";
import type {
  ConvergencePoint,
  LearningStatus,
  ProblemInstance,
  RunSummary,
  SolveJobPayload,
  SolverConfig,
  SolverResult,
  SystemInfo
} from "./types";

type AppTab = "solver" | "experiments" | "system" | "guide";

const PackingViewer = lazy(() =>
  import("./components/PackingViewer").then((module) => ({ default: module.PackingViewer }))
);

const FALLBACK_CONFIG: SolverConfig = {
  algorithm: "simulated_annealing",
  decoder: "extreme_points",
  configuration_id: "sa-default",
  placement_policy_id: "min_z_y_x",
  initial_temperature: null,
  target_initial_acceptance: 0.82,
  temperature_calibration_samples: 24,
  cooling_rate: 0.96,
  iterations_per_temperature: 80,
  min_temperature: 0.001,
  max_evaluations: 2500,
  time_limit_seconds: 10,
  stagnation_limit: 600,
  reheating_factor: 1.35,
  multi_start_count: 3,
  ruin_size: 4,
  use_native_core: false,
  parallel_mode: "sequential",
  parallel_workers: 1,
  convergence_sample_rate: 1,
  operator_probabilities: {
    swap: 0.32,
    relocate: 0.24,
    reverse: 0.14,
    rotate: 0.2,
    ruin_recreate: 0.1
  }
};

function prettyJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export default function App() {
  const [instanceJson, setInstanceJson] = useState("");
  const [config, setConfig] = useState<SolverConfig>(FALLBACK_CONFIG);
  const [seed, setSeed] = useState(123);
  const [persist, setPersist] = useState(true);
  const [result, setResult] = useState<SolverResult | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [learningStatus, setLearningStatus] = useState<LearningStatus | null>(null);
  const [solveJob, setSolveJob] = useState<SolveJobPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [tab, setTab] = useState<AppTab>("solver");
  const [monitorOpen, setMonitorOpen] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [livePoints, setLivePoints] = useState<ConvergencePoint[]>([]);

  const isJobActive = Boolean(solveJob && ["queued", "running", "paused"].includes(solveJob.status));
  const isSolving = Boolean(solveJob && ["queued", "running"].includes(solveJob.status));

  const parsedInstance = useMemo<ProblemInstance | null>(() => {
    if (!instanceJson.trim()) {
      return null;
    }
    try {
      return JSON.parse(instanceJson) as ProblemInstance;
    } catch {
      return null;
    }
  }, [instanceJson]);

  const refreshRuns = useCallback(async () => {
    try {
      const [nextRuns, nextLearning] = await Promise.all([fetchRuns(), fetchLearningStatus()]);
      setRuns(nextRuns);
      setLearningStatus(nextLearning);
    } catch {
      setRuns([]);
    }
  }, []);

  const loadDefaults = useCallback(async () => {
    setError(null);
    setNotice(null);
    try {
      const [instance, defaultConfig] = await Promise.all([fetchDefaultInstance(), fetchDefaultConfig()]);
      setInstanceJson(prettyJson(instance));
      setConfig(defaultConfig);
      setResult(null);
      setRunId(null);
      await refreshRuns();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao carregar defaults");
    }
  }, [refreshRuns]);

  useEffect(() => {
    void loadDefaults();
  }, [loadDefaults]);

  useEffect(() => {
    void fetchSystemInfo()
      .then(setSystemInfo)
      .catch(() => setSystemInfo(null));
  }, []);

  useEffect(() => {
    const jobId = solveJob?.job_id;
    const jobStatus = solveJob?.status;
    if (!jobId || !jobStatus || !["queued", "running", "paused"].includes(jobStatus)) {
      return;
    }
    const timer = window.setInterval(() => {
      void fetchSolveJob(jobId)
        .then((nextJob) => {
          setSolveJob(nextJob);
          const latest = nextJob.progress.latest_point;
          if (latest) {
            setLivePoints((previous) => {
              const lastEvaluation = previous.length > 0 ? previous[previous.length - 1].evaluation : -1;
              return latest.evaluation > lastEvaluation ? [...previous, latest] : previous;
            });
          }
          if (nextJob.status === "completed" && nextJob.result) {
            setResult(nextJob.result);
            setRunId(nextJob.run_id);
            setNotice({ tone: "success", text: "Execucao concluida; analise e solucao atualizadas." });
            void refreshRuns();
          }
          if (nextJob.status === "cancelled") {
            setNotice({ tone: "error", text: "Execucao interrompida." });
          }
          if (nextJob.status === "error") {
            const message = nextJob.error ?? "Falha ao executar solver";
            setError(message);
            setNotice({ tone: "error", text: message });
          }
        })
        .catch((requestError) => {
          const message = requestError instanceof Error ? requestError.message : "Falha ao consultar execucao";
          setError(message);
          setNotice({ tone: "error", text: message });
        });
    }, 700);
    return () => window.clearInterval(timer);
  }, [refreshRuns, solveJob?.job_id, solveJob?.status]);

  const handleSolve = async () => {
    setError(null);
    setNotice(null);
    if (solveJob && ["queued", "running", "paused"].includes(solveJob.status)) {
      try {
        const nextJob =
          solveJob.status === "paused"
            ? await resumeSolveJob(solveJob.job_id)
            : solveJob.pause_supported
              ? await pauseSolveJob(solveJob.job_id)
              : await cancelSolveJob(solveJob.job_id);
        setSolveJob(nextJob);
        setNotice({
          tone: nextJob.status === "paused" ? "success" : "error",
          text:
            nextJob.status === "paused"
              ? "Execucao pausada. Clique em Retomar para continuar."
              : nextJob.status === "running"
                ? "Execucao retomada."
                : "Execucao interrompida."
        });
      } catch (requestError) {
        const message = requestError instanceof Error ? requestError.message : "Falha ao controlar execucao";
        setError(message);
        setNotice({ tone: "error", text: message });
      }
      return;
    }
    let instance: ProblemInstance;
    try {
      instance = JSON.parse(instanceJson) as ProblemInstance;
    } catch {
      setError("JSON inválido");
      return;
    }
    try {
      const job = await startSolveJob(instance, config, seed, persist);
      setSolveJob(job);
      setResult(null);
      setRunId(null);
      setLivePoints([]);
      setNotice({ tone: "success", text: "Execucao iniciada; acompanhando progresso em tempo real." });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao executar solver";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleCancelSolve = async () => {
    if (!solveJob || !["queued", "running", "paused"].includes(solveJob.status)) {
      return;
    }
    setError(null);
    try {
      const nextJob = await cancelSolveJob(solveJob.job_id);
      setSolveJob(nextJob);
      setNotice({ tone: "error", text: "Execucao interrompida." });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao interromper execucao";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleSuggestConfig = async () => {
    setError(null);
    setNotice(null);
    if (!parsedInstance) {
      setError("JSON inválido: corrija a instância antes de pedir sugestão");
      return;
    }
    try {
      const payload = await suggestConfig(parsedInstance, 1, seed);
      const suggestion = payload.suggestions[0];
      if (!suggestion) {
        setNotice({ tone: "error", text: "Nenhuma sugestão disponível." });
        return;
      }
      setConfig({ ...config, ...suggestion.config });
      if (payload.mode === "contextual_ego") {
        const predicted = suggestion.predicted_loss?.toFixed(4) ?? "?";
        const sigma = suggestion.uncertainty?.toFixed(4) ?? "?";
        setNotice({
          tone: "success",
          text: `Configuração sugerida pelo EGO contextual (${payload.observation_count} observações): perda prevista ${predicted} ± ${sigma}.`
        });
      } else {
        setNotice({
          tone: "success",
          text: `Aprendizado ainda aquecendo (${payload.observation_count} observações): sugestão aleatória do espaço de busca aplicada.`
        });
      }
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao sugerir configuração";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleImportSpreadsheet = async (file: File) => {
    setError(null);
    setNotice(null);
    try {
      const payload = await importSpreadsheet(file);
      setInstanceJson(prettyJson(payload.instance));
      setResult(null);
      setRunId(null);
      if (!payload.valid) {
        setError(`Planilha importada com avisos: ${payload.diagnostics.join("; ")}`);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao importar planilha");
    }
  };

  const handleDownloadTemplate = async () => {
    setError(null);
    setNotice(null);
    try {
      await downloadSpreadsheetTemplate();
      setNotice({ tone: "success", text: "Modelo de planilha gerado para download." });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao baixar modelo";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleExportXlsx = async () => {
    setError(null);
    setNotice(null);
    if (!parsedInstance) {
      const message = "JSON invalido: corrija a instancia antes de exportar";
      setError(message);
      setNotice({ tone: "error", text: message });
      return;
    }
    try {
      await exportInstanceSpreadsheet(parsedInstance);
      setNotice({ tone: "success", text: "Instancia exportada em planilha." });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao exportar planilha";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleExportResult = async () => {
    setError(null);
    setNotice(null);
    if (!parsedInstance || !result || result.placements.length === 0) {
      const message = "Execute o solver e encontre uma solucao antes de exportar o resultado";
      setError(message);
      setNotice({ tone: "error", text: message });
      return;
    }
    try {
      await exportResultSpreadsheet(parsedInstance, result, runId);
      setNotice({ tone: "success", text: "Planilha da solucao gerada com item, caixa, coordenadas e dimensoes." });
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Falha ao exportar resultado";
      setError(message);
      setNotice({ tone: "error", text: message });
    }
  };

  const handleSelectRun = async (nextRunId: number) => {
    setError(null);
    setNotice(null);
    try {
      const bundle = await fetchRunBundle(nextRunId);
      setInstanceJson(prettyJson(bundle.instance));
      setConfig(bundle.config);
      setResult(bundle.result);
      setRunId(nextRunId);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao carregar execução");
    }
  };

  const displayStatus = isJobActive ? solveJob?.status ?? "idle" : result?.status ?? "idle";

  const tabs: Array<{ id: AppTab; label: string; icon: React.ReactNode }> = [
    { id: "solver", label: "Solver", icon: <Boxes size={15} aria-hidden="true" /> },
    { id: "experiments", label: "Experimentos", icon: <FlaskConical size={15} aria-hidden="true" /> },
    { id: "system", label: "Sistema", icon: <MonitorCog size={15} aria-hidden="true" /> },
    { id: "guide", label: "Guia", icon: <BookOpen size={15} aria-hidden="true" /> }
  ];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-brand">
          <h1>Adaptive 3D Bin Packing</h1>
          <span>{parsedInstance?.instance_id ?? "offline_3d_bpp"}</span>
        </div>
        <nav className="app-tabs" aria-label="Seções da aplicação">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              className={tab === item.id ? "active" : ""}
              onClick={() => setTab(item.id)}
              aria-label={item.label}
              aria-current={tab === item.id ? "page" : undefined}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="topbar-right">
          <button
            type="button"
            className={`monitor-toggle${monitorOpen ? " active" : ""}`}
            onClick={() => setMonitorOpen((open) => !open)}
            title="Monitor de recursos flutuante (CPU/RAM em qualquer aba)"
            aria-pressed={monitorOpen}
          >
            <Activity size={15} aria-hidden="true" />
          </button>
          <div className="status-pill" data-status={displayStatus}>
            {displayStatus}
          </div>
        </div>
      </header>
      {notice ? (
        <div className="app-alert" data-tone={notice.tone} role="status" aria-live="polite">
          {notice.text}
        </div>
      ) : null}

      {tab === "solver" ? (
        <div className="workspace">
          <aside className="left-column">
            <SolverControls
              config={config}
              seed={seed}
              persist={persist}
              onConfigChange={setConfig}
              onSeedChange={setSeed}
              onPersistChange={setPersist}
              onSolve={handleSolve}
              onCancel={handleCancelSolve}
              onReset={loadDefaults}
              solveJob={solveJob}
              isJobActive={isJobActive}
              systemInfo={systemInfo}
              onSuggest={handleSuggestConfig}
            />
            <JsonEditor
              value={instanceJson}
              error={error}
              busy={isSolving}
              onChange={setInstanceJson}
              onImportFile={handleImportSpreadsheet}
              onDownloadTemplate={handleDownloadTemplate}
              onExportXlsx={handleExportXlsx}
            />
          </aside>

          <section className="center-column">
            <Suspense fallback={<div className="viewer-loading">Carregando 3D</div>}>
              <PackingViewer instance={parsedInstance} result={result} />
            </Suspense>
            <AnalysisWorkbench
              result={result}
              instance={parsedInstance}
              config={config}
              learningStatus={learningStatus}
              livePoints={isJobActive ? livePoints : undefined}
            />
          </section>

          <aside className="right-column">
            <MetricsRail result={result} runId={runId} onExportResult={handleExportResult} />
            <LearningPanel status={learningStatus} />
            <RunsPanel runs={runs} selectedRunId={runId} onSelectRun={handleSelectRun} />
          </aside>
        </div>
      ) : null}

      {tab === "experiments" ? (
        <ExperimentsLab instance={parsedInstance} baseConfig={config} baseSeed={seed} systemInfo={systemInfo} />
      ) : null}

      {tab === "system" ? <SystemPanel info={systemInfo} /> : null}

      {tab === "guide" ? <GuidePanel /> : null}

      {monitorOpen ? <FloatingMonitor onClose={() => setMonitorOpen(false)} /> : null}
    </main>
  );
}
