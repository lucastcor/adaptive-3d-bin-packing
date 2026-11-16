"""Exact Gaussian Process with Matern-5/2 ARD kernel and MAP hyperparameter fit.

Hyperparameters (ARD lengthscales and signal variance) are estimated by
maximising the log marginal likelihood (Rasmussen & Williams, 2006, ch. 5,
eq. 5.8/5.9) with analytic gradients and the Adam optimiser, plus a weak
log-normal prior on the log-hyperparameters for numerical robustness (MAP
estimation, as practised by Snoek et al., 2012, and the GPyTorch/BoTorch
defaults). Observation noise is supplied externally (heteroscedastic, one
variance per observation) and is correctly rescaled to the normalised target
space before being added to the kernel diagonal.

This implementation stays dependency-light (NumPy only) and can be replaced by
BoTorch/GPyTorch without changing configurator interfaces.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adaptive_3dbp.surrogates.base import Prediction

_SQRT5 = math.sqrt(5.0)
_LOG_PARAM_MIN = -5.0
_LOG_PARAM_MAX = 7.0
_PRIOR_STD = 3.0


class ExactGaussianProcess:
    def __init__(
        self,
        lengthscales: NDArray[np.float64] | None = None,
        signal_variance: float = 1.0,
        jitter: float = 1e-8,
    ) -> None:
        self.lengthscales = lengthscales
        self.signal_variance = signal_variance
        self.jitter = jitter
        self._x: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._x_mean: NDArray[np.float64] | None = None
        self._x_std: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._alpha: NDArray[np.float64] | None = None

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        noise: NDArray[np.float64] | None = None,
        optimize_hyperparameters: bool = True,
        max_iterations: int = 200,
        n_restarts: int = 2,
        seed: int = 0,
    ) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D matrix")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be a 1D vector with one target per row")

        self._x_mean = x.mean(axis=0)
        self._x_std = np.maximum(x.std(axis=0), 1e-9)
        x_norm = (x - self._x_mean) / self._x_std
        self._y_mean = float(y.mean())
        self._y_std = float(max(y.std(), 1e-9))
        y_norm = (y - self._y_mean) / self._y_std

        # Heteroscedastic noise is given in the original target scale; the GP
        # operates on normalised targets, so variances scale by 1/y_std^2.
        noise_norm: NDArray[np.float64] | None = None
        if noise is not None:
            noise_norm = np.maximum(noise, 0.0) / (self._y_std**2)

        if self.lengthscales is None:
            self.lengthscales = np.ones(x.shape[1], dtype=np.float64)

        if optimize_hyperparameters and x.shape[0] >= 4:
            self._optimize(x_norm, y_norm, noise_norm, max_iterations, n_restarts, seed)

        kernel = self._kernel(x_norm, x_norm)
        if noise_norm is not None:
            kernel = kernel + np.diag(np.maximum(noise_norm, self.jitter))
        kernel = kernel + np.eye(x.shape[0]) * self.jitter
        self._cholesky = np.linalg.cholesky(kernel)
        self._alpha = np.linalg.solve(self._cholesky.T, np.linalg.solve(self._cholesky, y_norm))
        self._x = x_norm
        self._y = y_norm

    def predict(self, x: NDArray[np.float64]) -> Prediction:
        if self._x is None or self._cholesky is None or self._alpha is None:
            raise RuntimeError("fit must be called before predict")
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("normalization state is missing")

        x_norm = (x - self._x_mean) / self._x_std
        k_star = self._kernel(x_norm, self._x)
        mean_norm = k_star @ self._alpha
        v = np.linalg.solve(self._cholesky, k_star.T)
        variance = np.maximum(self.signal_variance - np.sum(v * v, axis=0), self.jitter)
        return Prediction(mean=mean_norm * self._y_std + self._y_mean, std=np.sqrt(variance) * self._y_std)

    def log_marginal_likelihood(self) -> float:
        if self._y is None or self._cholesky is None or self._alpha is None:
            raise RuntimeError("fit must be called before log_marginal_likelihood")
        n = self._y.shape[0]
        return float(
            -0.5 * float(self._y @ self._alpha)
            - float(np.sum(np.log(np.diag(self._cholesky))))
            - 0.5 * n * math.log(2.0 * math.pi)
        )

    # ------------------------------------------------------------------
    # Hyperparameter optimisation (MAP via Adam, analytic gradients)
    # ------------------------------------------------------------------

    def _optimize(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        noise: NDArray[np.float64] | None,
        max_iterations: int,
        n_restarts: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        dims = x.shape[1]
        assert self.lengthscales is not None
        starts = [np.concatenate([np.log(self.lengthscales), [math.log(max(self.signal_variance, 1e-9))]])]
        for _ in range(max(0, n_restarts)):
            starts.append(rng.normal(0.0, 1.0, size=dims + 1))

        best_theta: NDArray[np.float64] | None = None
        best_objective = -np.inf
        for theta0 in starts:
            theta, objective = self._adam(x, y, noise, theta0, max_iterations)
            if objective > best_objective:
                best_objective = objective
                best_theta = theta

        if best_theta is not None:
            self.lengthscales = np.exp(best_theta[:dims])
            self.signal_variance = float(np.exp(best_theta[dims]))

    def _adam(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        noise: NDArray[np.float64] | None,
        theta0: NDArray[np.float64],
        max_iterations: int,
        learning_rate: float = 0.08,
    ) -> tuple[NDArray[np.float64], float]:
        theta = np.clip(theta0.copy(), _LOG_PARAM_MIN, _LOG_PARAM_MAX)
        first_moment = np.zeros_like(theta)
        second_moment = np.zeros_like(theta)
        best_theta = theta.copy()
        best_objective = -np.inf
        for iteration in range(1, max_iterations + 1):
            objective, gradient = self._map_objective_and_grad(x, y, noise, theta)
            if objective > best_objective:
                best_objective = objective
                best_theta = theta.copy()
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            m_hat = first_moment / (1.0 - 0.9**iteration)
            v_hat = second_moment / (1.0 - 0.999**iteration)
            theta = np.clip(
                theta + learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8),
                _LOG_PARAM_MIN,
                _LOG_PARAM_MAX,
            )
            if float(np.max(np.abs(gradient))) < 1e-5:
                break
        return best_theta, best_objective

    def _map_objective_and_grad(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        noise: NDArray[np.float64] | None,
        theta: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        dims = x.shape[1]
        n = x.shape[0]
        lengthscales = np.exp(theta[:dims])
        signal_variance = float(np.exp(theta[dims]))

        scaled = x / lengthscales
        diff = scaled[:, None, :] - scaled[None, :, :]
        radius = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        envelope = np.exp(-_SQRT5 * radius)
        poly = 1.0 + _SQRT5 * radius + 5.0 * radius * radius / 3.0
        kernel_signal = signal_variance * poly * envelope

        kernel = kernel_signal + np.eye(n) * self.jitter
        if noise is not None:
            kernel = kernel + np.diag(np.maximum(noise, self.jitter))

        try:
            cholesky = np.linalg.cholesky(kernel)
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros_like(theta)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, y))
        log_likelihood = (
            -0.5 * float(y @ alpha)
            - float(np.sum(np.log(np.diag(cholesky))))
            - 0.5 * n * math.log(2.0 * math.pi)
        )
        # Weak log-normal prior over log-hyperparameters (MAP regularisation).
        log_prior = -0.5 * float(np.sum((theta / _PRIOR_STD) ** 2))
        objective = log_likelihood + log_prior

        kernel_inverse = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, np.eye(n))
        )
        trace_term = np.outer(alpha, alpha) - kernel_inverse  # (alpha alpha^T - K^-1)

        gradient = np.zeros_like(theta)
        # d k / d log lengthscale_k = (5/3) * sv * (1 + sqrt5 r) * exp(-sqrt5 r) * (diff_k^2)
        common = (5.0 / 3.0) * signal_variance * (1.0 + _SQRT5 * radius) * envelope
        for k in range(dims):
            dk = common * (diff[:, :, k] ** 2)
            gradient[k] = 0.5 * float(np.sum(trace_term * dk))
        # d k / d log signal_variance = kernel_signal
        gradient[dims] = 0.5 * float(np.sum(trace_term * kernel_signal))
        gradient -= theta / (_PRIOR_STD**2)
        return objective, gradient

    # ------------------------------------------------------------------

    def _kernel(self, left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.lengthscales is None:
            raise RuntimeError("lengthscales are not initialized")
        scaled_left = left / self.lengthscales
        scaled_right = right / self.lengthscales
        diff = scaled_left[:, None, :] - scaled_right[None, :, :]
        radius = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        return self.signal_variance * (1.0 + _SQRT5 * radius + 5.0 * radius * radius / 3.0) * np.exp(-_SQRT5 * radius)

    def save(self, path: Path) -> None:
        if self._x is None or self._y is None or self._cholesky is None or self._alpha is None:
            raise RuntimeError("fit must be called before save")
        if self._x_mean is None or self._x_std is None or self.lengthscales is None:
            raise RuntimeError("normalization state is missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self._x,
            y=self._y,
            x_mean=self._x_mean,
            x_std=self._x_std,
            y_mean=np.asarray([self._y_mean], dtype=np.float64),
            y_std=np.asarray([self._y_std], dtype=np.float64),
            lengthscales=self.lengthscales,
            signal_variance=np.asarray([self.signal_variance], dtype=np.float64),
            jitter=np.asarray([self.jitter], dtype=np.float64),
            cholesky=self._cholesky,
            alpha=self._alpha,
        )

    @classmethod
    def load(cls, path: Path) -> "ExactGaussianProcess":
        data = np.load(path)
        gp = cls(
            lengthscales=data["lengthscales"],
            signal_variance=float(data["signal_variance"][0]),
            jitter=float(data["jitter"][0]),
        )
        gp._x = data["x"]
        gp._y = data["y"]
        gp._x_mean = data["x_mean"]
        gp._x_std = data["x_std"]
        gp._y_mean = float(data["y_mean"][0])
        gp._y_std = float(data["y_std"][0])
        gp._cholesky = data["cholesky"]
        gp._alpha = data["alpha"]
        return gp
