import { Database } from "lucide-react";
import type { RunSummary } from "../types";

type RunsPanelProps = {
  runs: RunSummary[];
  selectedRunId: number | null;
  onSelectRun: (runId: number) => void;
};

export function RunsPanel({ runs, selectedRunId, onSelectRun }: RunsPanelProps) {
  return (
    <section className="panel runs-panel" aria-label="Execuções salvas">
      <div className="panel-heading">
        <Database size={18} aria-hidden="true" />
        <h2>Runs</h2>
      </div>
      <div className="runs-list">
        {runs.map((run) => (
          <button
            className={selectedRunId === run.run_id ? "run-row active" : "run-row"}
            key={run.run_id}
            type="button"
            onClick={() => onSelectRun(run.run_id)}
            title={`Carregar run ${run.run_id}`}
          >
            <strong>#{run.run_id}</strong>
            <span>{run.instance_id}</span>
            <span>{run.bins_used}/{run.lower_bound}</span>
          </button>
        ))}
        {runs.length === 0 ? <span className="muted">-</span> : null}
      </div>
    </section>
  );
}
