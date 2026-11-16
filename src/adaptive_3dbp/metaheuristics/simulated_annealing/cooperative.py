"""Central-memory cooperation channel for parallel multi-start SA.

Design (see docs/decisoes_cientificas.md for references):

- Cooperation model: *central memory / solution warehouse* (Crainic, 2019;
  Crainic & Toulouse, 2010). Workers run independent SA chains and exchange
  only the best-so-far solution through a shared store; there is no fine
  grained shared state.
- Event ordering: a monotonically increasing ``version`` counter updated under
  a lock acts as a logical clock (Lamport, 1978). Workers decide "is this
  knowledge new to me?" by comparing versions, never wall-clock timestamps,
  so no physical clock synchronization (NTP/Berkeley) is needed: on a single
  host all processes already share one clock source, and even distributed,
  ordering by logical version is sufficient for this exchange pattern.
- Failure handling: workers emit heartbeats; the coordinator (parent process)
  detects stalled workers by heartbeat timeout and degrades gracefully
  (drops their contribution). A bully-style election (Garcia-Molina, 1982) is
  unnecessary here because the coordinator is fixed: if the parent dies the
  whole computation dies with it. Election protocols become relevant only in
  a future multi-node deployment.
"""

from __future__ import annotations

import time
from typing import Any

from adaptive_3dbp.domain.models import CandidateSolution


class CooperativeChannel:
    def __init__(self, state: Any, lock: Any, worker_id: int, sync_interval: int) -> None:
        self.state = state
        self.lock = lock
        self.worker_id = worker_id
        self.sync_interval = max(1, sync_interval)
        self.last_seen_version = 0

    def heartbeat(self) -> None:
        self.state[f"heartbeat_{self.worker_id}"] = time.time()

    def publish(self, score: tuple[float, ...], candidate: CandidateSolution) -> bool:
        """Store the candidate as the global best if it improves it (lexicographic)."""
        score_tuple = tuple(float(value) for value in score)
        with self.lock:
            current = self.state.get("best_score")
            if current is None or score_tuple < tuple(current):
                self.state["best_score"] = score_tuple
                self.state["best_order"] = tuple(candidate.item_order)
                self.state["best_rotations"] = tuple(candidate.rotations)
                self.state["best_policy"] = candidate.placement_policy_id
                self.state["best_worker"] = self.worker_id
                self.state["version"] = int(self.state.get("version", 0)) + 1
                return True
        return False

    def report_evaluations(self, evaluations: int) -> None:
        """Expose per-worker progress so the coordinator can report it."""
        self.state[f"evaluations_{self.worker_id}"] = int(evaluations)

    def is_cancelled(self) -> bool:
        return bool(self.state.get("control_cancelled", False))

    def wait_if_paused(self) -> float:
        """Block while the coordinator signals pause; returns seconds spent paused.

        The pause flag travels through the same central memory used for
        solution exchange, so no extra IPC channel is needed. Heartbeats keep
        flowing during the pause so the worker is not flagged as stalled.
        """
        if not self.state.get("control_paused", False):
            return 0.0
        started = time.perf_counter()
        while self.state.get("control_paused", False):
            if self.is_cancelled():
                break
            self.heartbeat()
            time.sleep(0.1)
        return time.perf_counter() - started

    def fetch_better_than(self, score: tuple[float, ...]) -> CandidateSolution | None:
        """Return the shared best candidate when it is new and strictly better."""
        with self.lock:
            version = int(self.state.get("version", 0))
            if version == self.last_seen_version:
                return None
            self.last_seen_version = version
            best_score = self.state.get("best_score")
            if best_score is None:
                return None
            if not tuple(best_score) < tuple(float(value) for value in score):
                return None
            return CandidateSolution(
                item_order=tuple(self.state["best_order"]),
                rotations=tuple(self.state["best_rotations"]),
                placement_policy_id=str(self.state["best_policy"]),
            )
