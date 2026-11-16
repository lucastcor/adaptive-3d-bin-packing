import { LineChart } from "lucide-react";
import { useMemo, useState } from "react";
import type { ConvergencePoint } from "../types";

type ConvergenceStripProps = {
  points: ConvergencePoint[];
};

type Mode = "best" | "current" | "bins" | "acceptance";

const WIDTH = 460;
const HEIGHT = 110;
const MAX_SAMPLES = 600;

type SamplePoint = {
  evaluation: number;
  value: number;
  improved: boolean;
};

function formatValue(value: number, mode: Mode) {
  if (mode === "acceptance") {
    return `${Math.round(value * 100)}%`;
  }
  if (mode === "bins") {
    return String(Math.round(value));
  }
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toLocaleString("pt-BR", { maximumFractionDigits: 3 })} bi`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toLocaleString("pt-BR", { maximumFractionDigits: 2 })} mi`;
  }
  if (abs >= 10_000) {
    return `${(value / 1_000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mil`;
  }
  if (abs >= 100) {
    return value.toLocaleString("pt-BR", { maximumFractionDigits: 0 });
  }
  return value.toLocaleString("pt-BR", { maximumFractionDigits: 3 });
}

function formatDelta(first: number, last: number, mode: Mode) {
  if (Math.abs(first - last) <= Math.max(Math.abs(first), 1) * 1e-9 && mode !== "acceptance") {
    return "sem variação";
  }
  if (mode === "acceptance") {
    return `${Math.round(last * 100)}%`;
  }
  const delta = first - last;
  if (mode === "bins") {
    return delta === 0 ? "0" : `${delta > 0 ? "-" : "+"}${Math.abs(Math.round(delta))}`;
  }
  if (Math.abs(first) < 1e-9) {
    return formatValue(delta, mode);
  }
  const percent = (delta / Math.abs(first)) * 100;
  return `${percent >= 0 ? "-" : "+"}${Math.abs(percent).toLocaleString("pt-BR", { maximumFractionDigits: 2 })}%`;
}

export function ConvergenceStrip({ points }: ConvergenceStripProps) {
  const [mode, setMode] = useState<Mode>("best");

  const chart = useMemo(() => {
    if (points.length === 0) {
      return null;
    }
    let accepted = 0;
    const raw: SamplePoint[] = points.map((point, index) => {
      if (point.accepted) {
        accepted += 1;
      }
      let value = point.best_objective_scalar ?? point.objective_scalar;
      if (mode === "current") {
        value = point.current_objective_scalar ?? point.objective_scalar;
      }
      if (mode === "bins") {
        value = point.best_bins ?? point.bins_used;
      }
      if (mode === "acceptance") {
        value = accepted / Math.max(index + 1, 1);
      }
      return {
        evaluation: point.evaluation,
        value,
        improved: point.improved
      };
    }).filter((point) => Number.isFinite(point.value));

    if (raw.length === 0) {
      return null;
    }

    // Downsample long histories while preserving first, last and improvement events.
    const stride = Math.max(1, Math.ceil(raw.length / MAX_SAMPLES));
    const sampled: SamplePoint[] = [];
    for (let index = 0; index < raw.length; index += 1) {
      const point = raw[index];
      if (index === 0 || index === raw.length - 1 || point.improved || index % stride === 0) {
        sampled.push(point);
      }
    }
    if (sampled[sampled.length - 1] !== raw[raw.length - 1]) {
      sampled.push(raw[raw.length - 1]);
    }

    // Min/max from the data itself (forcing min=0/max=1 flattened every curve).
    let min = Infinity;
    let max = -Infinity;
    for (const point of sampled) {
      if (point.value < min) min = point.value;
      if (point.value > max) max = point.value;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return null;
    }
    const first = raw[0].value;
    const span = max - min;
    const isFlat = span <= Math.max(Math.abs(max), 1) * 1e-9;
    const pad = span > 0 ? span * 0.12 : Math.max(Math.abs(max) * 0.1, 0.5);
    min -= pad;
    max += pad;
    const range = max - min;
    const minEvaluation = sampled[0].evaluation;
    const maxEvaluation = sampled[sampled.length - 1].evaluation;
    const evaluationRange = Math.max(maxEvaluation - minEvaluation, 1);

    const coords = sampled.map((point) => {
      const x = sampled.length <= 1 ? WIDTH / 2 : ((point.evaluation - minEvaluation) / evaluationRange) * WIDTH;
      const y = HEIGHT - ((point.value - min) / range) * HEIGHT;
      return [x, y] as const;
    });
    const line = coords
      .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
      .join(" ");
    const firstX = coords[0][0];
    const [lastX, lastY] = coords[coords.length - 1];
    const area = `${line} L ${lastX.toFixed(2)} ${HEIGHT} L ${firstX.toFixed(2)} ${HEIGHT} Z`;
    const last = sampled[sampled.length - 1].value;

    return {
      line,
      area,
      min: min + pad,
      max: max - pad,
      first,
      last,
      lastX,
      lastY,
      isFlat,
      sampledCount: sampled.length
    };
  }, [mode, points]);

  const acceptedCount = points.filter((point) => point.accepted).length;
  const improvedCount = points.filter((point) => point.improved).length;
  const acceptedRatio = points.length ? acceptedCount / points.length : 0;
  const lastEvaluation = points.length ? points[points.length - 1].evaluation : 0;
  const flatLabel = chart?.isFlat
    ? mode === "bins"
      ? `${formatValue(chart.last, mode)} caixa${Math.round(chart.last) === 1 ? "" : "s"}`
      : mode === "acceptance"
        ? `${formatValue(chart.last, mode)} aceite`
        : "sem variação"
    : null;

  return (
    <section className="panel convergence-panel" aria-label="Convergência">
      <div className="convergence-heading">
        <div className="panel-heading">
          <LineChart size={18} aria-hidden="true" />
          <h2>Convergência</h2>
        </div>
        <div className="segmented compact chart-mode" role="group" aria-label="Métrica da curva">
          <button type="button" className={mode === "best" ? "active" : ""} onClick={() => setMode("best")}>
            Melhor
          </button>
          <button type="button" className={mode === "current" ? "active" : ""} onClick={() => setMode("current")}>
            Atual
          </button>
          <button type="button" className={mode === "bins" ? "active" : ""} onClick={() => setMode("bins")}>
            Caixas
          </button>
          <button type="button" className={mode === "acceptance" ? "active" : ""} onClick={() => setMode("acceptance")}>
            Aceite
          </button>
        </div>
      </div>
      {chart ? (
        <div className="chart-frame">
          <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="Curva de convergência">
            <defs>
              <linearGradient id="curve-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2563eb" stopOpacity="0.22" />
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0.02" />
              </linearGradient>
            </defs>
            {[0.25, 0.5, 0.75].map((fraction) => (
              <line
                key={fraction}
                x1="0"
                x2={WIDTH}
                y1={HEIGHT * fraction}
                y2={HEIGHT * fraction}
                className="gridline"
              />
            ))}
            <path d={chart.area} fill="url(#curve-fill)" stroke="none" />
            <path d={chart.line} className="curve" vectorEffect="non-scaling-stroke" />
            <circle cx={chart.lastX} cy={chart.lastY} r="3.5" className="curve-dot" />
          </svg>
          <div className="chart-scale" data-flat={chart.isFlat ? "true" : "false"}>
            <span>{chart.isFlat ? "" : formatValue(chart.max, mode)}</span>
            <span className="chart-last">{flatLabel ?? formatValue(chart.last, mode)}</span>
            <span>{chart.isFlat ? "" : formatValue(chart.min, mode)}</span>
          </div>
        </div>
      ) : (
        <div className="chart-empty">Execute o solver para ver a curva de convergência.</div>
      )}
      <div className="convergence-stats">
        <span>Aval. {lastEvaluation.toLocaleString("pt-BR")}</span>
        <span>{points.length.toLocaleString("pt-BR")} pontos</span>
        <span>{Math.round(acceptedRatio * 100)}% aceite</span>
        <span>{improvedCount} melhorias</span>
        {chart ? <span>Delta {formatDelta(chart.first, chart.last, mode)}</span> : null}
      </div>
    </section>
  );
}
