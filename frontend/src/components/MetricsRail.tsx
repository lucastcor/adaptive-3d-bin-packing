import { Activity, Box, FileDown, Gauge, Timer } from "lucide-react";
import { InfoTip } from "./InfoTip";
import type { SolverResult } from "../types";

type MetricsRailProps = {
  result: SolverResult | null;
  runId: number | null;
  onExportResult: () => void;
};

function formatNumber(value: number, digits = 3) {
  return new Intl.NumberFormat("pt-BR", {
    maximumFractionDigits: digits
  }).format(value);
}

export function MetricsRail({ result, runId, onExportResult }: MetricsRailProps) {
  const canExportResult = Boolean(result && result.placements.length > 0);

  return (
    <section className="panel metrics-panel" aria-label="Métricas">
      <div className="panel-heading split-heading">
        <div className="heading-label">
          <Activity size={18} aria-hidden="true" />
          <h2>Métricas</h2>
          <InfoTip text="Resumo da solução encontrada, tempo, limite inferior teórico e execução salva." />
        </div>
        <button
          className="ghost-button compact-button result-export-button"
          type="button"
          onClick={onExportResult}
          disabled={!canExportResult}
          title="Baixar planilha com a posição final de cada item"
        >
          <FileDown size={15} aria-hidden="true" />
          <span>Baixar solução</span>
        </button>
      </div>

      <div className="metric-grid">
        <div className="metric">
          <Box size={18} aria-hidden="true" />
          <span>Caixas</span>
          <strong>{result ? result.best_found_bins : "-"}</strong>
        </div>
        <div className="metric">
          <Gauge size={18} aria-hidden="true" />
          <span>
            Limite inf.
            <InfoTip text="Mínimo teórico de caixas. Se Caixas = Limite inf., a solução atingiu esse limite." />
          </span>
          <strong>{result ? result.lower_bound : "-"}</strong>
        </div>
        <div className="metric">
          <Timer size={18} aria-hidden="true" />
          <span>Tempo</span>
          <strong>{result ? `${formatNumber(result.runtime_seconds, 2)}s` : "-"}</strong>
        </div>
        <div className="metric">
          <Activity size={18} aria-hidden="true" />
          <span>Aval.</span>
          <strong>{result ? formatNumber(result.evaluations, 0) : "-"}</strong>
        </div>
      </div>

      <dl className="detail-list">
        <div>
          <dt>Status</dt>
          <dd>{result?.status ?? "-"}</dd>
        </div>
        <div>
          <dt title="Distância percentual entre a solução encontrada e o limite inferior teórico.">Gap limite</dt>
          <dd>{result ? `${formatNumber(result.gap_to_lower_bound * 100, 2)}%` : "-"}</dd>
        </div>
        <div>
          <dt>Solver</dt>
          <dd>{result?.solver ?? "-"}</dd>
        </div>
        <div>
          <dt>Run</dt>
          <dd>{runId ?? "-"}</dd>
        </div>
      </dl>

      {result && result.bin_metrics.length > 0 ? (
        <div className="bin-table" aria-label="Métricas por caixa">
          {result.bin_metrics.map((metric) => (
            <div className="bin-row" key={metric.bin_id}>
              <span>Bin {metric.bin_id + 1}</span>
              <meter min={0} max={1} value={metric.utilization} />
              <strong>{formatNumber(metric.utilization * 100, 1)}%</strong>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
