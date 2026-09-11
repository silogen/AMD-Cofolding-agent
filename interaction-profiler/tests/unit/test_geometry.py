"""Unit tests for geometry kernels (no heavy deps beyond numpy)."""

from __future__ import annotations

import numpy as np

from interaction_profiler.interactions.geometry import (
    angle_deg,
    pairwise_distances,
    plane_angle_deg,
    ring_offset,
    vector_angle_deg,
)


def test_pairwise_distances_matches_bruteforce():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    d = pairwise_distances(a, b)
    assert d.shape == (2, 2)
    assert abs(d[0, 0] - 0.0) < 1e-9
    assert abs(d[0, 1] - 5.0) < 1e-9
    assert abs(d[1, 0] - 1.0) < 1e-9


def test_angle_deg_right_angle():
    p1 = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    assert abs(angle_deg(p1, vertex, p3) - 90.0) < 1e-6


def test_angle_deg_linear():
    p1 = np.array([-1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    assert abs(angle_deg(p1, vertex, p3) - 180.0) < 1e-6


def test_ring_offset():
    centroid = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    # point directly above the centroid -> zero in-plane offset
    assert abs(ring_offset(np.array([0.0, 0.0, 3.5]), centroid, normal)) < 1e-9
    # point shifted 2 A in-plane -> offset 2.0 regardless of height
    assert abs(ring_offset(np.array([2.0, 0.0, 3.5]), centroid, normal) - 2.0) < 1e-9


def test_vector_and_plane_angle():
    u = np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 0.0, -1.0])
    assert abs(vector_angle_deg(u, v) - 180.0) < 1e-6
    # plane angle folds to 0-90
    assert abs(plane_angle_deg(u, v) - 0.0) < 1e-6
