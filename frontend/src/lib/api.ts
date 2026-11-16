import type {
  LearningStatus,
  ProblemInstance,
  RunBundle,
  RunSummary,
  SolveJobPayload,
  SolvePayload,
  SolverConfig,
  SolverResult,
  SystemInfo,
  SystemUsage
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers
    },
    ...init
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchDefaultInstance(): Promise<ProblemInstance> {
  return request<ProblemInstance>("/api/examples/default-instance");
}

export function fetchDefaultConfig(): Promise<SolverConfig> {
  return request<SolverConfig>("/api/config/default");
}

export function solveInstance(
  instance: ProblemInstance,
  config: SolverConfig,
  seed: number,
  persist: boolean
): Promise<SolvePayload> {
  return request<SolvePayload>("/api/solve", {
    method: "POST",
    body: JSON.stringify({ instance, config, seed, persist })
  });
}

export function startSolveJob(
  instance: ProblemInstance,
  config: SolverConfig,
  seed: number,
  persist: boolean
): Promise<SolveJobPayload> {
  return request<SolveJobPayload>("/api/solve-jobs", {
    method: "POST",
    body: JSON.stringify({ instance, config, seed, persist })
  });
}

export function fetchSolveJob(jobId: string): Promise<SolveJobPayload> {
  return request<SolveJobPayload>(`/api/solve-jobs/${jobId}`);
}

export function pauseSolveJob(jobId: string): Promise<SolveJobPayload> {
  return request<SolveJobPayload>(`/api/solve-jobs/${jobId}/pause`, { method: "POST" });
}

export function resumeSolveJob(jobId: string): Promise<SolveJobPayload> {
  return request<SolveJobPayload>(`/api/solve-jobs/${jobId}/resume`, { method: "POST" });
}

export function cancelSolveJob(jobId: string): Promise<SolveJobPayload> {
  return request<SolveJobPayload>(`/api/solve-jobs/${jobId}/cancel`, { method: "POST" });
}

export function fetchRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/api/runs?limit=10");
}

export function fetchRunBundle(runId: number): Promise<RunBundle> {
  return request<RunBundle>(`/api/runs/${runId}/bundle`);
}

export function fetchLearningStatus(): Promise<LearningStatus> {
  return request<LearningStatus>("/api/learning/status");
}

export function fetchSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>("/api/system/info");
}

export function fetchSystemUsage(): Promise<SystemUsage> {
  return request<SystemUsage>("/api/system/usage");
}

export type ConfigSuggestion = {
  config: SolverConfig;
  predicted_loss: number | null;
  uncertainty: number | null;
  expected_improvement: number | null;
};

export type SuggestPayload = {
  mode: "contextual_ego" | "random_fallback";
  observation_count: number;
  incumbent_predicted_loss?: number;
  reason?: string;
  suggestions: ConfigSuggestion[];
};

export function suggestConfig(instance: ProblemInstance, nSuggestions = 1, seed = 0): Promise<SuggestPayload> {
  return request<SuggestPayload>("/api/config/suggest", {
    method: "POST",
    body: JSON.stringify({ instance, n_suggestions: nSuggestions, seed })
  });
}

export type CampaignArm = {
  arm: string;
  status: string;
  tuning_seconds: number;
  tuning_solver_runs: number;
  training_best_loss: number | null;
  validation_loss: number | null;
  test_loss: number | null;
  validation_metrics?: CampaignMetrics | null;
  test_metrics?: CampaignMetrics | null;
  reason: string | null;
};

export type CampaignMetrics = {
  runs: number;
  feasible_runs: number;
  timeout_runs: number;
  error_runs: number;
  optimal_runs: number;
  optimal_rate: number | null;
  mean_loss: number | null;
  mean_bins_used: number | null;
  best_bins_used: number | null;
  mean_gap_to_lower_bound: number | null;
  mean_runtime_seconds: number | null;
  mean_evaluations: number | null;
  mean_bin_utilization: number | null;
};

export type CampaignPayload = {
  job_id: string;
  status: "queued" | "running" | "completed" | "error";
  progress: {
    stage: string;
    detail: string;
    fraction: number;
    completed_solver_runs?: number;
    estimated_solver_runs?: number;
    current_arm?: string | null;
    current_phase?: string | null;
  };
  summary: {
    campaign_id: string;
    elapsed_seconds: number;
    records: number;
    estimated_solver_runs?: number;
    dataset?: {
      source: "generated" | "manifest" | string;
      manifest_path: string;
      summary: Record<string, number>;
      per_class: Record<string, number>;
      classes: string[];
      curation?: {
        instances: number;
        accepted: number;
        fallback: number;
        class_replacements?: number;
        class_slots?: number;
        class_slots_replaced?: number;
        discriminative?: number;
        too_easy: number;
        too_slow: number;
        timeouts: number;
        errors: number;
        mean_preflight_evaluations?: number;
        mean_preflight_lower_bound?: number;
        mean_preflight_bins_spread?: number;
        mean_preflight_gap_spread?: number;
      } | null;
    };
    arms: CampaignArm[];
    report_markdown?: string;
    comparison_valid?: boolean;
    comparison_warning?: string;
    dataset_discrimination_warning?: string;
    time_budget_saturated?: boolean;
    time_budget_warning?: string;
    artifacts: Record<string, string | null>;
  } | null;
  error: string | null;
};

export function startCampaign(spec: Record<string, unknown>): Promise<CampaignPayload> {
  return request<CampaignPayload>("/api/campaigns", {
    method: "POST",
    body: JSON.stringify(spec)
  });
}

export function fetchCampaign(jobId: string): Promise<CampaignPayload> {
  return request<CampaignPayload>(`/api/campaigns/${jobId}`);
}

export type CampaignArtifact =
  | "runs.csv"
  | "solutions.jsonl"
  | "report.md"
  | "metadata.json"
  | "dataset_manifest.json"
  | "summary.json"
  | "comparison.png"
  | "results_table.tex";

export function campaignFileUrl(jobId: string, artifact: CampaignArtifact): string {
  return `${API_BASE}/api/campaigns/${jobId}/files/${artifact}`;
}

export async function downloadCampaignFile(jobId: string, artifact: CampaignArtifact): Promise<void> {
  const response = await fetch(campaignFileUrl(jobId, artifact));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  triggerDownload(await response.blob(), artifact);
}

export type SpreadsheetImportPayload = {
  valid: boolean;
  diagnostics: string[];
  instance: ProblemInstance;
};

export async function importSpreadsheet(file: File): Promise<SpreadsheetImportPayload> {
  const response = await fetch(`${API_BASE}/api/instances/import-spreadsheet`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return response.json() as Promise<SpreadsheetImportPayload>;
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function downloadSpreadsheetTemplate(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/examples/spreadsheet-template`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  triggerDownload(await response.blob(), "instance_template.xlsx");
}

export async function exportInstanceSpreadsheet(instance: ProblemInstance): Promise<void> {
  const response = await fetch(`${API_BASE}/api/instances/export-spreadsheet`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(instance)
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  triggerDownload(await response.blob(), `${instance.instance_id || "instance"}.xlsx`);
}

export async function exportResultSpreadsheet(
  instance: ProblemInstance,
  result: SolverResult,
  runId: number | null
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/results/export-spreadsheet`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance, result, run_id: runId })
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  triggerDownload(await response.blob(), `${instance.instance_id || "solution"}_solution.xlsx`);
}
