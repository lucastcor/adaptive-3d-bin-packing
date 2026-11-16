"""Benchmarking harness to compare in-house configurators against SMAC3 and irace.

SMAC3 (Lindauer et al., JMLR 2022) and irace (Lopez-Ibanez et al., ORP 2016)
are the de-facto baselines for automated algorithm configuration. SMAC3 is an
optional Python dependency (``pip install .[tuners]``); irace runs in R and is
driven through generated scenario files.
"""

from adaptive_3dbp.tuning.objective import TuningObjective

__all__ = ["TuningObjective"]
