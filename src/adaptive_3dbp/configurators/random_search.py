from __future__ import annotations

import random

from adaptive_3dbp.configurators.search_space import SearchSpace
from adaptive_3dbp.domain.models import SolverConfig


class RandomSearchConfigurator:
    name = "random_search"

    def __init__(self, search_space: SearchSpace, seed: int = 0) -> None:
        self.search_space = search_space
        self.rng = random.Random(seed)

    def suggest(self, n_suggestions: int) -> list[SolverConfig]:
        return [self.search_space.sample(self.rng) for _ in range(n_suggestions)]

