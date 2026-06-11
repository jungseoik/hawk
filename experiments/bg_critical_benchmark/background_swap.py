"""
E2 — Counterfactual Background-Swap Compositor + BSI metric (Route B)
====================================================================
Holds the moving FOREGROUND fixed and swaps the static BACKGROUND, to test
causally whether a model uses the static stream.

Pipeline (run when data is available):
  1. extract foreground from a source clip via the SAME Farneback motion mask
     used by the model (hawk/processors/video_processor.py), so the swap is
     consistent with the model's decomposition;
  2. composite foreground onto {safe_bg, dangerous_bg};
  3. run the model on each composite;
  4. BSI = 1 - sentence_sim(resp_safe, resp_dangerous).

This file ships the compositor + BSI scaffold. The model call is left as a hook
(`run_model`) so it can be wired to app.py / a checkpoint later. A synthetic
self-test verifies the compositing + BSI math today (no model, no data).

Dependencies: numpy (required); opencv-python (`--full` tier) for real frames.
"""
from __future__ import annotations

import argparse
import numpy as np

MAG_THRESHOLD = 0.2


def motion_mask(prev_gray, cur_gray):
    """Same flow + threshold as the model's decomposition."""
    import cv2
    fl = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
    return (mag > MAG_THRESHOLD).astype(np.uint8)


def composite(foreground_rgb, background_rgb, mask):
    """mask⊙foreground + (1-mask)⊙background  (exact complementary partition)."""
    m = mask[..., None]
    return m * foreground_rgb + (1 - m) * background_rgb


def bsi(resp_safe: str, resp_danger: str, embed_fn=None) -> float:
    """
    Background Sensitivity Index = 1 - sim(safe_response, dangerous_response).
    High => model output changes with background => uses the static stream.
    Default sim is token-Jaccard (no deps); pass embed_fn for sentence cosine.
    """
    if embed_fn is not None:
        a, b = embed_fn(resp_safe), embed_fn(resp_danger)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return 1.0 - cos
    sa, sb = set(resp_safe.lower().split()), set(resp_danger.lower().split())
    jac = len(sa & sb) / (len(sa | sb) + 1e-12)
    return 1.0 - jac


def run_model(frames):  # pragma: no cover - hook
    """Wire to the trained CERBERUS model later (returns a description string)."""
    raise NotImplementedError("Wire to checkpoint via app.py inference when available.")


def _self_test():
    print("=" * 60)
    print("E2 self-test — compositor partition + BSI math")
    print("=" * 60)
    rng = np.random.default_rng(0)
    fg = (rng.random((64, 64, 3)) * 255)
    safe = np.zeros((64, 64, 3)) + 30
    danger = np.zeros((64, 64, 3)) + 200
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 1  # moving foreground region

    comp_safe = composite(fg, safe, mask)
    comp_danger = composite(fg, danger, mask)
    # foreground (mask==1) must be identical across swaps; only background differs
    fg_ok = np.allclose(comp_safe[mask == 1], comp_danger[mask == 1])
    bg_diff = not np.allclose(comp_safe[mask == 0], comp_danger[mask == 0])
    print(f"foreground preserved across swap : {fg_ok}")
    print(f"background differs across swap   : {bg_diff}")

    b_blind = bsi("a person walks on a road", "a person walks on a road")
    b_aware = bsi("a person walks on a sidewalk", "a person stands on a highway danger")
    print(f"BSI (background-blind model)  = {b_blind:.3f}  (expected ~0)")
    print(f"BSI (background-aware model)  = {b_aware:.3f}  (expected >0)")
    print("-" * 60)
    print(f"RESULT: {'PASS' if fg_ok and bg_diff and b_aware > b_blind else 'CHECK'}")


def main():
    ap = argparse.ArgumentParser(description="E2 background-swap compositor / BSI")
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()
    _self_test()


if __name__ == "__main__":
    main()
