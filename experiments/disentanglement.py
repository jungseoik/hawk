"""
E1 — Disentanglement Measurement for Complementary Visual Decomposition (CVD)
=============================================================================
Paper experiment E1 (docs/cerberus-research-plan.md §3).

Goal: quantify whether the dynamic (z_m) and static (z_b) representation streams
are *complementary* (non-redundant) while their union still covers the
task-relevant information carried by the full-frame appearance stream (z_a).

This module provides:
  - linear_cka / rbf_cka      : representational similarity (redundancy proxy)
  - gaussian_mi               : closed-form MI estimate for (approx) Gaussian reps
  - knn_mi                    : Kraskov-Stögbauer-Grassberger kNN MI estimator
  - cosine_stats              : distribution of pairwise cosine similarities
  - complementary_disentanglement_score (CDS)  : the headline metric (NEW)

CDS (Complementary Disentanglement Score), proposed in this work:
    Coverage   = CKA( [z_m ; z_b] , z_a )      # union aligns with full-frame info
    Redundancy = CKA( z_m , z_b )              # overlap between the two streams
    CDS        = Coverage * (1 - Redundancy)   # high  <=> covered AND non-redundant

Runs with no dataset: the __main__ self-test fabricates synthetic representations
that mimic complementary / redundant / collapsed regimes and shows CDS separates
them. On a trained model, feed real representations dumped by
`scripts/extract_representations.py`.

Dependencies: numpy (required); torch/sklearn optional.
"""
from __future__ import annotations

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Centered Kernel Alignment (CKA)
# ---------------------------------------------------------------------------
def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA in [0, 1]. X: (n, d1), Y: (n, d2). Higher => more redundant."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx, Ky = X @ X.T, Y @ Y.T
    hsic = np.sum(_center_gram(Kx) * _center_gram(Ky))
    nx = np.sqrt(np.sum(_center_gram(Kx) * _center_gram(Kx)))
    ny = np.sqrt(np.sum(_center_gram(Ky) * _center_gram(Ky)))
    denom = nx * ny
    return float(hsic / denom) if denom > 0 else 0.0


def _rbf_gram(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    sq = np.sum(X * X, 1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0)
    if sigma is None:  # median heuristic
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        sigma = np.sqrt(med / 2) if med > 0 else 1.0
    return np.exp(-d2 / (2 * sigma ** 2))


def rbf_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF (nonlinear) CKA in [0, 1]."""
    Kx, Ky = _rbf_gram(X), _rbf_gram(Y)
    hsic = np.sum(_center_gram(Kx) * _center_gram(Ky))
    nx = np.sqrt(np.sum(_center_gram(Kx) * _center_gram(Kx)))
    ny = np.sqrt(np.sum(_center_gram(Ky) * _center_gram(Ky)))
    denom = nx * ny
    return float(hsic / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Mutual information estimates
# ---------------------------------------------------------------------------
def gaussian_mi(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """
    Closed-form MI (nats) assuming jointly Gaussian X, Y.
    I = 0.5 * log( det(Cx) det(Cy) / det(C[xy]) ).
    Reduce dimensionality (e.g. PCA) before calling on high-d reps for stability.
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    n = X.shape[0]
    Z = np.concatenate([X, Y], 1)
    dx = X.shape[1]
    Cx = (X.T @ X) / (n - 1) + eps * np.eye(dx)
    Cy = (Y.T @ Y) / (n - 1) + eps * np.eye(Y.shape[1])
    Cz = (Z.T @ Z) / (n - 1) + eps * np.eye(Z.shape[1])
    _, ldx = np.linalg.slogdet(Cx)
    _, ldy = np.linalg.slogdet(Cy)
    _, ldz = np.linalg.slogdet(Cz)
    return float(max(0.5 * (ldx + ldy - ldz), 0.0))


def knn_mi(X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
    """
    Kraskov-Stögbauer-Grassberger (KSG) kNN MI estimator (nats), estimator 1.
    Falls back gracefully; requires scipy. Use for non-Gaussian reps.
    """
    try:
        from scipy.spatial import cKDTree
        from scipy.special import digamma
    except Exception as e:  # pragma: no cover
        raise RuntimeError("knn_mi requires scipy") from e
    n = X.shape[0]
    X = X + 1e-10 * np.random.randn(*X.shape)
    Y = Y + 1e-10 * np.random.randn(*Y.shape)
    Z = np.concatenate([X, Y], 1)
    tree_z = cKDTree(Z)
    # distance to k-th neighbor in joint space (Chebyshev / inf-norm)
    eps = tree_z.query(Z, k=k + 1, p=np.inf)[0][:, k]
    tx, ty = cKDTree(X), cKDTree(Y)
    nx = np.array([len(tx.query_ball_point(X[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(Y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# Cosine similarity distribution
# ---------------------------------------------------------------------------
def cosine_stats(X: np.ndarray, Y: np.ndarray) -> dict:
    """Per-sample cosine similarity between paired rows of X and Y."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(Xn * Yn, 1)
    return {"mean": float(cos.mean()), "std": float(cos.std()),
            "p10": float(np.percentile(cos, 10)), "p90": float(np.percentile(cos, 90))}


# ---------------------------------------------------------------------------
# CDS — the headline metric
# ---------------------------------------------------------------------------
def complementary_disentanglement_score(
    z_a: np.ndarray, z_m: np.ndarray, z_b: np.ndarray, kernel: str = "linear"
) -> dict:
    """
    Complementary Disentanglement Score.

    Coverage   = CKA([z_m ; z_b], z_a)   in [0,1]   (union covers full-frame info)
    Redundancy = CKA(z_m, z_b)           in [0,1]   (overlap of the two streams)
    CDS        = Coverage * (1 - Redundancy)        in [0,1], higher is better.
    """
    cka = linear_cka if kernel == "linear" else rbf_cka
    union = np.concatenate([z_m, z_b], 1)
    coverage = cka(union, z_a)
    redundancy = cka(z_m, z_b)
    cds = coverage * (1.0 - redundancy)
    return {"coverage": coverage, "redundancy": redundancy, "cds": cds}


def full_report(z_a, z_m, z_b) -> dict:
    """Convenience: full E1 diagnostic bundle for one set of representations."""
    rep = {
        "cka_motion_bg": linear_cka(z_m, z_b),
        "rbf_cka_motion_bg": rbf_cka(z_m, z_b),
        "cka_app_motion": linear_cka(z_a, z_m),
        "gaussian_mi_motion_bg": gaussian_mi(z_m, z_b),
        "cosine_motion_bg": cosine_stats(z_m, z_b),
    }
    rep.update(complementary_disentanglement_score(z_a, z_m, z_b))
    return rep


# ---------------------------------------------------------------------------
# Synthetic self-test (no dataset, no trained model required)
# ---------------------------------------------------------------------------
def _make_synthetic(regime: str, n: int = 512, d: int = 64, seed: int = 0):
    """
    Fabricate (z_a, z_m, z_b) for three regimes to validate that CDS behaves:
      - 'complementary' : z_m, z_b independent; union spans z_a   -> high CDS
      - 'partial'       : z_m, z_b share some factors             -> mid  CDS
      - 'collapsed'     : z_m ~= z_b (redundant, e.g. buggy loss) -> low  CDS
    """
    rng = np.random.default_rng(seed)
    fm = rng.standard_normal((n, d))          # dynamic latent factors
    fb = rng.standard_normal((n, d))          # static latent factors
    if regime == "complementary":
        z_m, z_b = fm, fb
    elif regime == "partial":
        shared = rng.standard_normal((n, d))
        z_m = 0.7 * fm + 0.7 * shared
        z_b = 0.7 * fb + 0.7 * shared
    elif regime == "collapsed":
        z_m = fm
        z_b = fm + 0.05 * rng.standard_normal((n, d))
    else:
        raise ValueError(regime)
    # appearance = full-frame info ~ union of dynamic + static factors
    W = rng.standard_normal((2 * d, d))
    z_a = np.concatenate([fm, fb], 1) @ W + 0.1 * rng.standard_normal((n, d))
    return z_a, z_m, z_b


def _self_test():
    print("=" * 68)
    print("E1 self-test — CDS separates complementary vs collapsed regimes")
    print("=" * 68)
    for regime in ["complementary", "partial", "collapsed"]:
        z_a, z_m, z_b = _make_synthetic(regime)
        rep = complementary_disentanglement_score(z_a, z_m, z_b)
        print(f"[{regime:13s}]  coverage={rep['coverage']:.3f}  "
              f"redundancy={rep['redundancy']:.3f}  CDS={rep['cds']:.3f}")
    print("-" * 68)
    print("Expected ordering:  CDS(complementary) > CDS(partial) > CDS(collapsed)")


def main():
    ap = argparse.ArgumentParser(description="E1 disentanglement metrics")
    ap.add_argument("--reps", type=str, default=None,
                    help="npz with arrays z_a, z_m, z_b (from extract_representations.py)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.reps:
        data = np.load(args.reps)
        rep = full_report(data["z_a"], data["z_m"], data["z_b"])
        for k, v in rep.items():
            print(f"{k:24s}: {v}")
    else:
        _self_test()


if __name__ == "__main__":
    main()
