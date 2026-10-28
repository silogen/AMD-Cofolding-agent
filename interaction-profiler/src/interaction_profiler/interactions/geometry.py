# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vectorized distance / angle kernels.

Pure-numpy by default (fast for pocket-sized problems). When ``use_gpu``
is requested and Torch is importable, the pairwise distance matrix is
computed with ``torch.cdist`` on the default CUDA/ROCm device -- useful
when profiling large batches where the O(n_lig * n_rec) matrix dominates.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_TORCH = None
_TORCH_TRIED = False


def _torch():
    global _TORCH, _TORCH_TRIED
    if not _TORCH_TRIED:
        _TORCH_TRIED = True
        try:
            import torch

            _TORCH = torch
        except Exception:
            _TORCH = None
    return _TORCH


def gpu_available() -> bool:
    t = _torch()
    try:
        return bool(t is not None and t.cuda.is_available())
    except Exception:
        return False


def pairwise_distances(a: np.ndarray, b: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Return the (len(a), len(b)) Euclidean distance matrix."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.empty((len(a), len(b)), dtype=np.float64)

    if use_gpu:
        t = _torch()
        if t is not None and t.cuda.is_available():
            try:
                ta = t.as_tensor(a, device="cuda", dtype=t.float32)
                tb = t.as_tensor(b, device="cuda", dtype=t.float32)
                d = t.cdist(ta, tb)
                return d.cpu().numpy().astype(np.float64)
            except Exception as exc:
                logger.debug("GPU cdist failed, falling back to numpy: %s", exc)

    # numpy: ||a||^2 + ||b||^2 - 2 a.b
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = aa + bb - 2.0 * a @ b.T
    np.clip(d2, 0.0, None, out=d2)
    return np.sqrt(d2)


def angle_deg(p1: np.ndarray, vertex: np.ndarray, p3: np.ndarray) -> float:
    """Angle p1-vertex-p3 in degrees."""
    v1 = np.asarray(p1, float) - np.asarray(vertex, float)
    v2 = np.asarray(p3, float) - np.asarray(vertex, float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def vector_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees (0-180)."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def ring_offset(point: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> float:
    """Lateral offset of ``point`` from a ring's centre, measured in the ring
    plane (the in-plane component of point-centroid). Used to require that a
    stacking ring / cation actually sits over the ring, not merely close."""
    v = np.asarray(point, float) - np.asarray(centroid, float)
    n = np.asarray(normal, float)
    nn = np.linalg.norm(n)
    if nn == 0:
        return float(np.linalg.norm(v))
    n = n / nn
    in_plane = v - np.dot(v, n) * n
    return float(np.linalg.norm(in_plane))


def plane_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle between two ring planes given their normals (0-90)."""
    a = vector_angle_deg(n1, n2)
    if a != a:  # nan
        return a
    return a if a <= 90.0 else 180.0 - a


__all__ = [
    "pairwise_distances",
    "angle_deg",
    "vector_angle_deg",
    "ring_offset",
    "plane_angle_deg",
    "gpu_available",
]
