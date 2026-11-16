import { BrainCircuit } from "lucide-react";
import type { LearningStatus } from "../types";

type LearningPanelProps = {
  status: LearningStatus | null;
};

export function LearningPanel({ status }: LearningPanelProps) {
  const latest = status?.latest_model;
  return (
    <section className="panel learning-panel" aria-label="Aprendizado online">
      <div className="panel-heading">
        <BrainCircuit size={18} aria-hidden="true" />
        <h2>GPR Online</h2>
      </div>
      <dl className="detail-list compact-details">
        <div>
          <dt>Observações</dt>
          <dd>{status?.observation_count ?? "-"}</dd>
        </div>
        <div>
          <dt>Modelo</dt>
          <dd>{latest ? `#${latest.model_id}` : "aquecendo"}</dd>
        </div>
        <div>
          <dt>Tipo</dt>
          <dd>{latest?.model_type ?? "-"}</dd>
        </div>
      </dl>
    </section>
  );
}
