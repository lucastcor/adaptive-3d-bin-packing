from __future__ import annotations

import numpy as np

from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


def test_exact_gp_predicts_training_shape() -> None:
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
    gp = ExactGaussianProcess()
    gp.fit(x, y)
    prediction = gp.predict(x)
    assert prediction.mean.shape == (3,)
    assert prediction.std.shape == (3,)
    assert np.all(prediction.std >= 0)

