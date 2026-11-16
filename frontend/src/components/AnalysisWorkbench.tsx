import {
  BrainCircuit,
  CheckCircle2,
  FlaskConical,
  Info,
  LineChart,
  ListTree,
  Thermometer,
  XCircle
} from "lucide-react";
import { useMemo, useState, type MouseEvent } from "react";
import type { ConvergencePoint, LearningStatus, ProblemInstance, SolverConfig, SolverResult } from "../types";

type AnalysisWorkbenchProps = {
  result: SolverResult | null;
  instance: ProblemInstance | null;
  config: SolverConfig;
  learningStatus: LearningStatus | null;
  livePoints?: ConvergencePoint[];
};

type AnalysisTab = "curve" | "sa" | "learning";
type ChartMode = "objective" | "bins" | "acceptance" | "temperature";

type TracePoint = {
  index: number;
  evaluation: number;
  elapsedSeconds: number;
  bestObjective: number;
  currentObjective: number;
  bestBins: number;
  currentBins: number;
  acceptanceRate: number;
  accepted: boolean;
  improved: boolean;
  temperature: number;
  delta: number | null;
};

type SeriesSpec = {
  key: string;
  label: string;
  color: string;
  value: (point: TracePoint) => number | null;
};

const CHART_WIDTH = 720;
const CHART_HEIGHT = 188;
const PLOT = { left: 58, right: 16, top: 14, bottom: 30 };
const PLOT_WIDTH = CHART_WIDTH - PLOT.left - PLOT.right;
const PLOT_HEIGHT = CHART_HEIGHT - PLOT.top - PLOT.bottom;
const MAX_PATH_POINTS = 900;

const modeCopy: Record<ChartMode, { title: string; help: string; axis: string }> = {
  objective: {
    title: "Score composto",
    help: "Energia E = caixas + r: a parte inteira e o numero de caixas; o decimal e a fragmentacao r = 1 - media(u²), menor quando o espaco livre se concentra numa caixa. Quanto menor, melhor.",
    axis: "score"
  },
  bins: {
    title: "Caixas usadas",
    help: "Mostra quantas caixas a melhor solucao e a solucao corrente usavam ao longo da busca.",
    axis: "caixas"
  },
  acceptance: {
    title: "Taxa de aceite",
    help: "No SA, movimentos piores podem ser aceitos no inicio para escapar de otimos locais.",
    axis: "aceite"
  },
  temperature: {
    title: "Temperatura",
    help: "A temperatura controla a chance de aceitar pioras. Conforme resfria, a busca fica mais seletiva.",
    axis: "temperatura"
  }
};

const operatorLabels: Record<string, string> = {
  swap: "troca",
  relocate: "realoca",
  reverse: "inverte",
  rotate: "rotacao",
  ruin_recreate: "ruin/recreate"
};

function formatNumber(value: number, digits = 2) {
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: digits }).format(value);
}

function formatCompact(value: number, digits = 2) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) {
    return `${formatNumber(value / 1_000_000_000, digits)} bi`;
  }
  if (abs >= 1_000_000) {
    return `${formatNumber(value / 1_000_000, digits)} mi`;
  }
  if (abs >= 10_000) {
    return `${formatNumber(value / 1_000, 1)} mil`;
  }
  if (abs >= 100) {
    return formatNumber(value, 0);
  }
  return formatNumber(value, digits);
}

function formatValue(value: number | null | undefined, mode: ChartMode) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  if (mode === "acceptance") {
    return `${formatNumber(value * 100, 1)}%`;
  }
  if (mode === "bins") {
    return formatNumber(value, 0);
  }
  if (mode === "objective") {
    return formatNumber(value, 4);
  }
  return formatCompact(value, mode === "temperature" ? 3 : 2);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function downsample(points: TracePoint[]) {
  if (points.length <= MAX_PATH_POINTS) {
    return points;
  }
  const stride = Math.ceil(points.length / MAX_PATH_POINTS);
  return points.filter((point, index) => {
    return index === 0 || index === points.length - 1 || point.improved || index % stride === 0;
  });
}

function getSeries(mode: ChartMode): SeriesSpec[] {
  if (mode === "objective") {
    return [
      {
        key: "best",
        label: "melhor score",
        color: "#2563eb",
        value: (point) => point.bestObjective
      },
      {
        key: "current",
        label: "score atual",
        color: "#0f766e",
        value: (point) => point.currentObjective
      }
    ];
  }
  if (mode === "bins") {
    return [
      {
        key: "best-bins",
        label: "melhor caixas",
        color: "#2563eb",
        value: (point) => point.bestBins
      },
      {
        key: "current-bins",
        label: "caixas atuais",
        color: "#b45309",
        value: (point) => point.currentBins
      }
    ];
  }
  if (mode === "acceptance") {
    return [
      {
        key: "acceptance",
        label: "aceite acumulado",
        color: "#7c3aed",
        value: (point) => point.acceptanceRate
      }
    ];
  }
  return [
    {
      key: "temperature",
      label: "temperatura",
      color: "#dc2626",
      value: (point) => point.temperature
    }
  ];
}

function buildPath(points: TracePoint[], series: SeriesSpec, xFor: (point: TracePoint) => number, yFor: (value: number) => number) {
  return points
    .map((point, index) => {
      const value = series.value(point);
      const x = xFor(point);
      const y = yFor(value ?? 0);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function activeY(point: TracePoint, series: SeriesSpec[], yFor: (value: number) => number) {
  const firstValue = series.map((item) => item.value(point)).find((value) => value !== null && Number.isFinite(value));
  return yFor(firstValue ?? 0);
}

function InteractiveTrace({ points, objectiveScale }: { points: ConvergencePoint[]; objectiveScale: number }) {
  const [mode, setMode] = useState<ChartMode>("objective");
  const [activePoint, setActivePoint] = useState<TracePoint | null>(null);

  const trace = useMemo<TracePoint[]>(() => {
    let acceptedCount = 0;
    return points.map((point, index) => {
      if (point.accepted) {
        acceptedCount += 1;
      }
      return {
        index,
        evaluation: point.evaluation,
        elapsedSeconds: point.elapsed_seconds,
        bestObjective: (point.best_objective_scalar ?? point.objective_scalar) / objectiveScale,
        currentObjective: (point.current_objective_scalar ?? point.objective_scalar) / objectiveScale,
        bestBins: point.best_bins ?? point.bins_used,
        currentBins: point.current_bins ?? point.bins_used,
        acceptanceRate: acceptedCount / Math.max(index + 1, 1),
        accepted: point.accepted,
        improved: point.improved,
        temperature: point.temperature,
        delta: point.delta
      };
    });
  }, [points]);

  const chart = useMemo(() => {
    if (trace.length === 0) {
      return null;
    }
    const series = getSeries(mode);
    const minEvaluation = trace[0].evaluation;
    const maxEvaluation = trace[trace.length - 1].evaluation;
    const evaluationRange = Math.max(maxEvaluation - minEvaluation, 1);
    const allValues = trace.flatMap((point) =>
      series
        .map((item) => item.value(point))
        .filter((value): value is number => value !== null && Number.isFinite(value))
    );
    if (allValues.length === 0) {
      return null;
    }

    let minValue = Math.min(...allValues);
    let maxValue = Math.max(...allValues);
    if (mode === "acceptance") {
      minValue = 0;
      maxValue = 1;
    }
    if (mode === "bins") {
      minValue -= 0.35;
      maxValue += 0.35;
    }
    const span = maxValue - minValue;
    const pad = mode === "acceptance" ? 0 : span > 0 ? span * 0.12 : Math.max(Math.abs(maxValue) * 0.1, 0.5);
    const yMin = minValue - pad;
    const yMax = maxValue + pad;
    const yRange = Math.max(yMax - yMin, 1e-9);
    const sampled = downsample(trace);
    const xFor = (point: TracePoint) => PLOT.left + ((point.evaluation - minEvaluation) / evaluationRange) * PLOT_WIDTH;
    const yFor = (value: number) => PLOT.top + ((yMax - value) / yRange) * PLOT_HEIGHT;
    const paths = series.map((item) => ({
      ...item,
      d: buildPath(sampled, item, xFor, yFor)
    }));

    return {
      series,
      paths,
      xFor,
      yFor,
      minEvaluation,
      maxEvaluation,
      yMin,
      yMax
    };
  }, [mode, trace]);

  const handleMove = (event: MouseEvent<SVGSVGElement>) => {
    if (!chart || trace.length === 0) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = clamp(((event.clientX - rect.left) / Math.max(rect.width, 1)) * CHART_WIDTH, PLOT.left, CHART_WIDTH - PLOT.right);
    const ratio = (localX - PLOT.left) / PLOT_WIDTH;
    const targetEvaluation = chart.minEvaluation + ratio * Math.max(chart.maxEvaluation - chart.minEvaluation, 1);
    let nearest = trace[0];
    let nearestDistance = Math.abs(nearest.evaluation - targetEvaluation);
    for (const point of trace) {
      const distance = Math.abs(point.evaluation - targetEvaluation);
      if (distance < nearestDistance) {
        nearest = point;
        nearestDistance = distance;
      }
    }
    setActivePoint(nearest);
  };

  if (!chart) {
    return <div className="analysis-empty">Execute o solver para ver o histórico da busca.</div>;
  }

  const selectedCopy = modeCopy[mode];
  const activeX = activePoint ? chart.xFor(activePoint) : null;
  const activeTop = activePoint ? activeY(activePoint, chart.series, chart.yFor) : null;
  const acceptedCount = trace.filter((point) => point.accepted).length;
  const improvedCount = trace.filter((point) => point.improved).length;
  const lastPoint = trace[trace.length - 1];

  return (
    <div className="analysis-trace">
      <div className="analysis-controls">
        <div>
          <strong>{selectedCopy.title}</strong>
          <span>{selectedCopy.help}</span>
        </div>
        <div className="segmented compact analysis-mode" role="group" aria-label="Métrica do histórico">
          <button type="button" className={mode === "objective" ? "active" : ""} onClick={() => setMode("objective")}>
            Objetivo
          </button>
          <button type="button" className={mode === "bins" ? "active" : ""} onClick={() => setMode("bins")}>
            Caixas
          </button>
          <button type="button" className={mode === "acceptance" ? "active" : ""} onClick={() => setMode("acceptance")}>
            Aceite
          </button>
          <button type="button" className={mode === "temperature" ? "active" : ""} onClick={() => setMode("temperature")}>
            Temp.
          </button>
        </div>
      </div>

      <div className="analysis-chart-wrap">
        <svg
          className="analysis-chart"
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          role="img"
          aria-label={`Grafico de ${selectedCopy.axis} por avaliacao do solver`}
          onMouseMove={handleMove}
          onMouseLeave={() => setActivePoint(null)}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
            const y = PLOT.top + fraction * PLOT_HEIGHT;
            const labelValue = chart.yMax - fraction * (chart.yMax - chart.yMin);
            return (
              <g key={fraction}>
                <line x1={PLOT.left} x2={CHART_WIDTH - PLOT.right} y1={y} y2={y} className="analysis-gridline" />
                <text x={PLOT.left - 10} y={y + 4} className="analysis-axis-label" textAnchor="end">
                  {formatValue(labelValue, mode)}
                </text>
              </g>
            );
          })}
          {[0, 0.5, 1].map((fraction) => {
            const x = PLOT.left + fraction * PLOT_WIDTH;
            const evaluation = chart.minEvaluation + fraction * (chart.maxEvaluation - chart.minEvaluation);
            return (
              <g key={fraction}>
                <line x1={x} x2={x} y1={PLOT.top} y2={PLOT.top + PLOT_HEIGHT} className="analysis-gridline faint" />
                <text x={x} y={CHART_HEIGHT - 8} className="analysis-axis-label" textAnchor="middle">
                  {formatNumber(evaluation, 0)}
                </text>
              </g>
            );
          })}
          <line
            x1={PLOT.left}
            x2={CHART_WIDTH - PLOT.right}
            y1={PLOT.top + PLOT_HEIGHT}
            y2={PLOT.top + PLOT_HEIGHT}
            className="analysis-axis"
          />
          <line x1={PLOT.left} x2={PLOT.left} y1={PLOT.top} y2={PLOT.top + PLOT_HEIGHT} className="analysis-axis" />
          {chart.paths.map((path) => (
            <path key={path.key} d={path.d} className="analysis-line" style={{ stroke: path.color }} />
          ))}
          {trace
            .filter((point) => point.improved)
            .map((point) => (
              <circle
                key={`${point.evaluation}-${point.index}`}
                cx={chart.xFor(point)}
                cy={chart.yFor(mode === "bins" ? point.bestBins : point.bestObjective)}
                r="3.5"
                className="improvement-dot"
              />
            ))}
          {activePoint && activeX !== null && activeTop !== null ? (
            <g>
              <line x1={activeX} x2={activeX} y1={PLOT.top} y2={PLOT.top + PLOT_HEIGHT} className="hover-line" />
              {chart.series.map((series) => {
                const value = series.value(activePoint);
                if (value === null) {
                  return null;
                }
                return (
                  <circle
                    key={series.key}
                    cx={activeX}
                    cy={chart.yFor(value)}
                    r="4.5"
                    className="hover-dot"
                    style={{ stroke: series.color }}
                  />
                );
              })}
            </g>
          ) : null}
          <rect
            x={PLOT.left}
            y={PLOT.top}
            width={PLOT_WIDTH}
            height={PLOT_HEIGHT}
            fill="transparent"
            pointerEvents="all"
          />
        </svg>

        {activePoint && activeX !== null && activeTop !== null ? (
          <div
            className="chart-tooltip"
            style={{
              left: `${clamp((activeX / CHART_WIDTH) * 100, 18, 82)}%`,
              top: `${clamp((activeTop / CHART_HEIGHT) * 100, 18, 70)}%`
            }}
          >
            <strong>Avaliacao {activePoint.evaluation.toLocaleString("pt-BR")}</strong>
            <span>tempo {formatNumber(activePoint.elapsedSeconds, 3)}s</span>
            <span>score melhor {formatValue(activePoint.bestObjective, "objective")}</span>
            <span>score atual {formatValue(activePoint.currentObjective, "objective")}</span>
            <span>caixas {activePoint.bestBins}/{activePoint.currentBins}</span>
            <span>{activePoint.accepted ? "movimento aceito" : "movimento rejeitado"}</span>
            <span>{activePoint.improved ? "melhorou o melhor global" : "sem melhoria global"}</span>
            <span>temp. {formatValue(activePoint.temperature, "temperature")}</span>
            <span>delta {activePoint.delta === null ? "-" : formatCompact(activePoint.delta, 2)}</span>
          </div>
        ) : null}
      </div>

      <div className="chart-legend">
        {chart.series.map((series) => (
          <span key={series.key}>
            <i style={{ background: series.color }} />
            {series.label}
          </span>
        ))}
        <span>
          <i className="legend-improvement" />
          melhoria global
        </span>
      </div>

      <div className="analysis-kpis">
        <div>
          <span>Avaliacoes</span>
          <strong>{lastPoint.evaluation.toLocaleString("pt-BR")}</strong>
        </div>
        <div>
          <span>Aceitos</span>
          <strong>{formatNumber((acceptedCount / Math.max(trace.length, 1)) * 100, 1)}%</strong>
        </div>
        <div>
          <span>Melhorias</span>
          <strong>{improvedCount}</strong>
        </div>
        <div>
          <span>Temperatura final</span>
          <strong>{formatValue(lastPoint.temperature, "temperature")}</strong>
        </div>
      </div>
    </div>
  );
}

function SaEvents({ result, config, objectiveScale }: { result: SolverResult | null; config: SolverConfig; objectiveScale: number }) {
  if (!result) {
    return <div className="analysis-empty">Execute ou carregue uma run para ver os eventos internos do SA.</div>;
  }

  const points = result.convergence;
  const accepted = points.filter((point) => point.accepted).length;
  const rejected = Math.max(points.length - accepted, 0);
  const worseAccepted = points.filter((point) => point.accepted && (point.delta ?? 0) > 0).length;
  const improvements = points.filter((point) => point.improved);
  const first = points[0];
  const last = points[points.length - 1];
  const acceptedPercent = points.length ? (accepted / points.length) * 100 : 0;
  const rejectedPercent = points.length ? (rejected / points.length) * 100 : 0;

  return (
    <div className="sa-grid">
      <div className="sa-story">
        <div className="analysis-note">
          <Info size={15} aria-hidden="true" />
          <span>
            Avaliacao significa uma solucao candidata decodificada pelo solver. Iteracoes podem ser amostradas por
            <strong> convergence_sample_rate</strong>; por isso o grafico mostra os pontos registrados.
          </span>
        </div>
        <div className="storyline">
          <div>
            <span>Inicio</span>
            <strong>{first ? `${first.bins_used} caixas` : "-"}</strong>
            <small>temp. {first ? formatCompact(first.temperature, 3) : "-"}</small>
          </div>
          <div>
            <span>Melhorias</span>
            <strong>{improvements.length}</strong>
            <small>{worseAccepted} pioras aceitas</small>
          </div>
          <div>
            <span>Final</span>
            <strong>{result.best_found_bins} caixas</strong>
            <small>limite inf. {result.lower_bound}</small>
          </div>
        </div>
        <div className="decision-bars">
          <div>
            <span>Aceitos</span>
            <strong>{formatNumber(acceptedPercent, 1)}%</strong>
            <meter min={0} max={100} value={acceptedPercent} />
          </div>
          <div>
            <span>Rejeitados</span>
            <strong>{formatNumber(rejectedPercent, 1)}%</strong>
            <meter min={0} max={100} value={rejectedPercent} />
          </div>
        </div>
      </div>

      <div className="event-list">
        <div className="event-list-heading">
          <ListTree size={16} aria-hidden="true" />
          <strong>Ultimas melhorias</strong>
        </div>
        {improvements.length > 0 ? (
          improvements.slice(-6).map((point) => (
            <div className="event-row" key={`${point.evaluation}-${point.objective_scalar}`}>
              <span>Aval. {point.evaluation.toLocaleString("pt-BR")}</span>
              <strong>{formatValue((point.best_objective_scalar ?? point.objective_scalar) / objectiveScale, "objective")}</strong>
              <small>{point.best_bins ?? point.bins_used} caixa(s)</small>
            </div>
          ))
        ) : (
          <div className="event-row muted-row">Nenhuma melhoria global registrada nessa amostra.</div>
        )}
      </div>

      <div className="operator-panel">
        <div className="event-list-heading">
          <FlaskConical size={16} aria-hidden="true" />
          <strong>Operadores</strong>
        </div>
        {Object.entries(config.operator_probabilities).map(([name, probability]) => (
          <div className="operator-row" key={name}>
            <span>{operatorLabels[name] ?? name}</span>
            <meter min={0} max={1} value={probability} />
            <strong>{formatNumber(probability * 100, 1)}%</strong>
          </div>
        ))}
      </div>

      <div className="event-list">
        <div className="event-list-heading">
          <Thermometer size={16} aria-hidden="true" />
          <strong>Resfriamento</strong>
        </div>
        <div className="event-row">
          <span>Temperatura inicial</span>
          <strong>{first ? formatCompact(first.temperature, 3) : "-"}</strong>
        </div>
        <div className="event-row">
          <span>Temperatura final</span>
          <strong>{last ? formatCompact(last.temperature, 3) : "-"}</strong>
        </div>
        <div className="event-row">
          <span>Taxa</span>
          <strong>{formatNumber(config.cooling_rate, 4)}</strong>
        </div>
      </div>
    </div>
  );
}

function LearningEgo({ status }: { status: LearningStatus | null }) {
  const latest = status?.latest_model;
  const metrics = latest?.metrics ?? {};
  const observationCount = status?.observation_count ?? 0;
  const trained = Boolean(latest && metrics.trained !== false);
  const readyForEgo = trained && observationCount >= 3;

  return (
    <div className="learning-grid">
      <div className="learning-status-card">
        <div className="learning-badge" data-ready={readyForEgo ? "true" : "false"}>
          {readyForEgo ? <CheckCircle2 size={16} aria-hidden="true" /> : <XCircle size={16} aria-hidden="true" />}
          <span>{readyForEgo ? "GPR treinado" : "GPR aquecendo"}</span>
        </div>
        <dl className="learning-details">
          <div>
            <dt>Observacoes</dt>
            <dd>{observationCount}</dd>
          </div>
          <div>
            <dt>Modelo</dt>
            <dd>{latest ? `#${latest.model_id}` : "-"}</dd>
          </div>
          <div>
            <dt>Tipo</dt>
            <dd>{latest?.model_type ?? "-"}</dd>
          </div>
          <div>
            <dt>Target medio</dt>
            <dd>{typeof metrics.target_mean === "number" ? formatCompact(metrics.target_mean, 4) : "-"}</dd>
          </div>
          <div>
            <dt>Melhor target</dt>
            <dd>{typeof metrics.target_min === "number" ? formatCompact(metrics.target_min, 4) : "-"}</dd>
          </div>
          <div>
            <dt>Dispersao</dt>
            <dd>{typeof metrics.target_std === "number" ? formatCompact(metrics.target_std, 4) : "-"}</dd>
          </div>
        </dl>
      </div>

      <div className="ego-flow" aria-label="Fluxo GPR e EGO">
        <div className="flow-step done">
          <span>1</span>
          <strong>Execucao</strong>
          <small>resultado salvo</small>
        </div>
        <div className={observationCount > 0 ? "flow-step done" : "flow-step"}>
          <span>2</span>
          <strong>Observacao</strong>
          <small>features + config + target</small>
        </div>
        <div className={trained ? "flow-step done" : "flow-step"}>
          <span>3</span>
          <strong>GPR</strong>
          <small>media + incerteza</small>
        </div>
        <div className={readyForEgo ? "flow-step done" : "flow-step"}>
          <span>4</span>
          <strong>EGO</strong>
          <small>Expected Improvement</small>
        </div>
      </div>

      <div className="analysis-note learning-note">
        <Info size={15} aria-hidden="true" />
        <span>
          O GPR aprende desempenho, nao monta a caixa. O EGO usa a media prevista e a incerteza para escolher proximas
          configuracoes promissoras; target menor significa menos caixas acima do limite inferior. O tempo e
          reportado a parte e nao entra no target.
        </span>
      </div>

      <div className="target-box">
        <strong>Target aprendido</strong>
        <code>gap_limite + penalidade_falha</code>
      </div>
    </div>
  );
}

export function AnalysisWorkbench({ result, instance, config, learningStatus, livePoints }: AnalysisWorkbenchProps) {
  const [tab, setTab] = useState<AnalysisTab>("curve");
  const isLive = !result && (livePoints?.length ?? 0) > 0;
  const points = result?.convergence ?? livePoints ?? [];
  // Since 0.2 the energy is dimensionless (E = bins + r). Runs saved by 0.1
  // used b*M + r*V with M = 1000*V, so they are rescaled for display.
  const legacyEnergy = result?.solver_version?.startsWith("0.1.") ?? false;
  const objectiveScale = legacyEnergy
    ? Math.max((instance?.bin.length ?? 1) * (instance?.bin.width ?? 1) * (instance?.bin.height ?? 1) * 1000, 1)
    : 1;

  return (
    <section className="panel analysis-panel" aria-label="Analise da busca">
      <div className="analysis-header">
        <div className="panel-heading">
          <LineChart size={18} aria-hidden="true" />
          <h2>Analise</h2>
          {isLive ? <span className="live-badge">ao vivo</span> : null}
        </div>
        <div className="segmented compact analysis-tabs" role="tablist" aria-label="Abas de analise">
          <button type="button" className={tab === "curve" ? "active" : ""} onClick={() => setTab("curve")} role="tab">
            Curva
          </button>
          <button type="button" className={tab === "sa" ? "active" : ""} onClick={() => setTab("sa")} role="tab">
            SA
          </button>
          <button
            type="button"
            className={tab === "learning" ? "active" : ""}
            onClick={() => setTab("learning")}
            role="tab"
          >
            GPR/EGO
          </button>
        </div>
      </div>

      <div className="analysis-body">
        {tab === "curve" ? <InteractiveTrace points={points} objectiveScale={objectiveScale} /> : null}
        {tab === "sa" ? <SaEvents result={result} config={config} objectiveScale={objectiveScale} /> : null}
        {tab === "learning" ? <LearningEgo status={learningStatus} /> : null}
      </div>
    </section>
  );
}
