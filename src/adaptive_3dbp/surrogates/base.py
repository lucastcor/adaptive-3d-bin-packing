from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Prediction:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


class SurrogateModel(Protocol):
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], noise: NDArray[np.float64] | None = None) -> None:
        ...

    def predict(self, x: NDArray[np.float64]) -> Prediction:
        ...

