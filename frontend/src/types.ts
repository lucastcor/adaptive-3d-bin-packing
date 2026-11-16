export type BinSpec = {
  length: number;
  width: number;
  height: number;
  max_weight: number | null;
};

export type ItemSpec = {
  id: string;
  length: number;
  width: number;
  height: number;
  weight: number | null;
  allowed_rotations: "all_orthogonal" | "fixed" | number[];
};

export type ProblemInstance = {
  instance_id: string;
  problem_type: string;
  bin: BinSpec;
  items: ItemSpec[];
  constraints: {
    orthogonal_only: boolean;
    stability_required: boolean;
    support_ratio_min: number | null;
    min_distance: Array<{
      item_id_a: string;
      item_id_b: string;
      distance: number;
    }>;
  };
  best_known_bins?: number | null;
};

export type SolverConfig = {
  algorithm: string;
  decoder: string;
  configuration_id: string;
  placement_policy_id: string;
  initial_temperature: number | null;
  target_initial_acceptance: number;
  temperature_calibration_samples: number;
  cooling_rate: number;
  iterations_per_temperature: number;
  min_temperature: number;
  max_evaluations: number;
  time_limit_seconds: number;
  stagnation_limit: number;
  reheating_factor: number;
  multi_start_count: number;
  ruin_size: number;
  use_native_core: boolean;
  parallel_mode: "sequential" | "process_pool" | "cooperative_process_pool";
  parallel_workers: number;
  cooperative_sync_interval?: number;
  heartbeat_timeout_seconds?: number;
  convergence_sample_rate: number;
  operator_probabilities: Record<string, number>;
};

export type SystemInfo = {
  os: string;
  os_release: string;
  architecture: string;
  processor: string;
  python_version: string;
  cpu_logical: number;
  cpu_physical: number | null;
  total_memory_bytes: number | null;
  gpus: string[];
  gpu_used_by_solver: boolean;
  psutil_available: boolean;
  pause_signals_available: boolean;
  native_core_available?: boolean;
  workers: {
    recommended: number;
    maximum: number;
    note: string;
  };
};

export type SystemUsage = {
  available: boolean;
  reason?: string;
  cpu_percent?: number;
  cpu_per_core?: number[];
  memory_total_bytes?: number;
  memory_used_bytes?: number;
  memory_percent?: number;
  process_memory_bytes?: number;
  process_cpu_percent?: number;
  load_average?: number[] | null;
};

export type Placement = {
  item_id: string;
  item_index: number;
  bin_id: number;
  x: number;
  y: number;
  z: number;
  rotation_id: number;
  placed_length: number;
  placed_width: number;
  placed_height: number;
};

export type BinMetric = {
  bin_id: number;
  utilization: number;
  unused_volume: number;
  occupied_volume: number;
  compactness: number;
};

export type ConvergencePoint = {
  evaluation: number;
  elapsed_seconds: number;
  bins_used: number;
  objective_scalar: number;
  accepted: boolean;
  temperature: number;
  current_bins: number | null;
  current_objective_scalar: number | null;
  best_bins: number | null;
  best_objective_scalar: number | null;
  delta: number | null;
  improved: boolean;
};

export type SolverResult = {
  instance_id: string;
  solver: string;
  solver_version: string;
  configuration_id: string;
  seed: number;
  status: "feasible" | "infeasible" | "timeout" | "error";
  best_found_bins: number;
  lower_bound: number;
  best_known_bins: number | null;
  optimal_proven: boolean;
  gap_to_best_known: number | null;
  gap_to_lower_bound: number;
  runtime_seconds: number;
  evaluations: number;
  objective_scalar: number;
  placements: Placement[];
  bin_metrics: BinMetric[];
  convergence: ConvergencePoint[];
  features: Record<string, number>;
  diagnostics: string[];
  artifacts: Record<string, string>;
};

export type SolvePayload = {
  run_id: number | null;
  result: SolverResult;
};

export type SolveJobPayload = {
  job_id: string;
  status: "queued" | "running" | "paused" | "completed" | "cancelled" | "error";
  pause_supported: boolean;
  run_id: number | null;
  error: string | null;
  progress: {
    percent: number;
    elapsed_seconds: number;
    evaluation: number;
    max_evaluations: number;
    time_limit_seconds: number;
    latest_point: ConvergencePoint | null;
  };
  result: SolverResult | null;
};

export type RunSummary = {
  run_id: number;
  instance_id: string;
  solver: string;
  configuration_id: string;
  seed: number;
  status: string;
  bins_used: number;
  lower_bound: number;
  runtime_seconds: number;
  evaluations: number;
  created_at: string;
};

export type RunBundle = {
  instance: ProblemInstance;
  config: SolverConfig;
  result: SolverResult;
};

export type LearningStatus = {
  observation_count: number;
  latest_model: null | {
    model_id: number;
    model_type: string;
    artifact_path: string;
    observation_count: number;
    created_at: string;
    metrics: Record<string, unknown>;
  };
};
