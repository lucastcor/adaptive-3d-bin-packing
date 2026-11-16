from __future__ import annotations

import numpy as np

from adaptive_3dbp.configurators.classic_ego import plugin_incumbent
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


def _toy_data(n: int = 24, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(2.0 * x[:, 0]) + 0.1 * x[:, 1]
    return x, y


def test_mle_improves_log_marginal_likelihood() -> None:
    x, y = _toy_data()
    fixed = ExactGaussianProcess()
    fixed.fit(x, y, optimize_hyperparameters=False)
    optimized = ExactGaussianProcess()
    optimized.fit(x, y, optimize_hyperparameters=True, max_iterations=120, n_restarts=1)
    assert optimized.log_marginal_likelihood() >= fixed.log_marginal_likelihood() - 1e-6


def test_heteroscedastic_noise_is_rescaled_with_target_scale() -> None:
    # Targets with large scale: raw noise added to a normalised kernel used to
    # drown the signal entirely. After the fix the GP must track the trend.
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    y = 1000.0 * x[:, 0] + rng.normal(0.0, 1.0, size=20)
    noise = np.full(20, 1.0)
    gp = ExactGaussianProcess()
    gp.fit(x, y, noise, optimize_hyperparameters=False)
    prediction = gp.predict(x)
    residual = float(np.mean(np.abs(prediction.mean - y)))
    assert residual < 50.0


def test_plugin_incumbent_resists_lucky_outlier() -> None:
    # One lucky noisy observation far below the latent function must not drag
    # the incumbent all the way down (winner's curse mitigation).
    x = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = np.full(12, 5.0)
    y[4] = 1.0  # lucky outlier
    noise = np.full(12, 4.0)
    gp = ExactGaussianProcess()
    gp.fit(x, y, noise, optimize_hyperparameters=False)
    incumbent = plugin_incumbent(gp, x)
    assert incumbent > float(np.min(y))


def test_gradient_matches_finite_differences() -> None:
    x, y = _toy_data(n=10)
    gp = ExactGaussianProcess()
    gp._x_mean = x.mean(axis=0)
    gp._x_std = np.maximum(x.std(axis=0), 1e-9)
    x_norm = (x - gp._x_mean) / gp._x_std
    y_norm = (y - y.mean()) / max(y.std(), 1e-9)
    theta = np.asarray([0.3, -0.2, 0.1])
    _, gradient = gp._map_objective_and_grad(x_norm, y_norm, None, theta)
    eps = 1e-5
    for index in range(theta.shape[0]):
        theta_plus = theta.copy()
        theta_plus[index] += eps
        theta_minus = theta.copy()
        theta_minus[index] -= eps
        obj_plus, _ = gp._map_objective_and_grad(x_norm, y_norm, None, theta_plus)
        obj_minus, _ = gp._map_objective_and_grad(x_norm, y_norm, None, theta_minus)
        numeric = (obj_plus - obj_minus) / (2 * eps)
        assert abs(numeric - gradient[index]) < 1e-3


def _contextual_tuner(observations: int = 12):
    import json
    import random
    from pathlib import Path

    from adaptive_3dbp.configurators.noisy_contextual_ego import NoiseAwareContextualEGO
    from adaptive_3dbp.configurators.search_space import SearchSpace

    root = Path(__file__).resolve().parents[2]
    space = SearchSpace(json.loads((root / "configs/search_spaces/sa_search_space.json").read_text(encoding="utf-8")))
    tuner = NoiseAwareContextualEGO(space, seed=3, feature_names=("a", "b"))
    rng = random.Random(5)
    for index in range(observations):
        context = {"a": rng.uniform(0, 1), "b": rng.uniform(0, 1)}
        tuner.observe(context, space.sample(rng), [rng.uniform(0, 1) for _ in range(3)])
    return tuner


def test_contextual_recommend_fits_gp_once_and_matches_refit(monkeypatch) -> None:
    # The zero-shot policy queries a frozen history once per held-out
    # instance; the GP fit is deterministic, so it must run once and give the
    # same recommendations as refitting for every query.
    cached = _contextual_tuner()
    fits = []
    original_fit = ExactGaussianProcess.fit

    def counting_fit(self, *args, **kwargs):
        fits.append(1)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(ExactGaussianProcess, "fit", counting_fit)
    contexts = [{"a": 0.1, "b": 0.9}, {"a": 0.7, "b": 0.2}, {"a": 0.5, "b": 0.5}]
    recommended = [cached.recommend(context, pool_seed=index) for index, context in enumerate(contexts)]
    assert len(fits) == 1

    for index, context in enumerate(contexts):
        assert _contextual_tuner().recommend(context, pool_seed=index) == recommended[index]
    assert len(fits) == 1 + len(contexts)

    # A new observation invalidates the cache.
    cached.observe({"a": 0.3, "b": 0.3}, recommended[0], [0.5, 0.4, 0.6])
    cached.recommend(contexts[0], pool_seed=0)
    assert len(fits) == 2 + len(contexts)

    # Replacing the last observation (same length) also invalidates it.
    cached.observations.pop()
    cached.observe({"a": 0.8, "b": 0.1}, recommended[1], [0.2, 0.3, 0.1])
    cached.recommend(contexts[0], pool_seed=0)
    assert len(fits) == 3 + len(contexts)
