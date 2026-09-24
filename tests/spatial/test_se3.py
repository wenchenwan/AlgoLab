import numpy as np

from algolab.spatial.se3 import inv_transform, transform_points
from algolab.tools import random_transform


def test_inv_transform(rng):
    T = random_transform(seed=rng)
    np.testing.assert_allclose(T @ inv_transform(T), np.eye(4), atol=1e-12)


def test_transform_points_matches_homogeneous(rng):
    T = random_transform(seed=rng)
    pts = rng.normal(size=(10, 3))
    homo = np.hstack([pts, np.ones((10, 1))]) @ T.T
    np.testing.assert_allclose(transform_points(T, pts), homo[:, :3])
