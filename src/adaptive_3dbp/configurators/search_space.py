from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from adaptive_3dbp.domain.models import SolverConfig


@dataclass(frozen=True, slots=True)
class SearchSpace:
    definition: dict[str, Any]

    def sample(self, rng: random.Random) -> SolverConfig:
        values: dict[str, Any] = {}
        for name, spec in self.definition.items():
            if name == "operator_probabilities":
                operators = spec["operators"]
                weights = [rng.expovariate(1.0) for _ in operators]
                total = sum(weights)
                values[name] = {operator: weight / total for operator, weight in zip(operators, weights, strict=True)}
                continue
            if spec["type"] == "float":
                if spec.get("log"):
                    low = math.log(float(spec["low"]))
                    high = math.log(float(spec["high"]))
                    values[name] = math.exp(rng.uniform(low, high))
                else:
                    values[name] = rng.uniform(float(spec["low"]), float(spec["high"]))
            elif spec["type"] == "int":
                if spec.get("log"):
                    low = math.log(int(spec["low"]))
                    high = math.log(int(spec["high"]))
                    values[name] = int(round(math.exp(rng.uniform(low, high))))
                else:
                    values[name] = rng.randint(int(spec["low"]), int(spec["high"]))
            elif spec["type"] == "categorical":
                values[name] = rng.choice(spec["choices"])
        return SolverConfig(**values)

    def encode(self, config: SolverConfig) -> list[float]:
        encoded: list[float] = []
        for name, spec in self.definition.items():
            value = getattr(config, name)
            if name == "operator_probabilities":
                for operator in spec["operators"]:
                    encoded.append(float(value.get(operator, 0.0)))
            elif spec["type"] == "categorical":
                encoded.append(float(spec["choices"].index(value)))
            else:
                encoded.append(float(value))
        return encoded

