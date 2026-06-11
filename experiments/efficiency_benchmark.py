"""
Efficiency Benchmark — Unified vs. Separate Optical-Flow Computation
====================================================================
Validates the §P4 / Analysis-D claim: computing optical flow ONCE and splitting
it into motion + background masks is ~2x cheaper than the naive implementation
that computes Farneback flow separately for the motion loader and the background
loader.

  separate : flow(motion) + flow(background)  -> 2 dense flow computations
  unified  : flow(once) -> motion_mask, bg_mask = 1 - motion_mask

Mirrors hawk/processors/video_processor.py::compute_motion_and_background.

Runs on a real video (--video) or on synthetic frames (default), so the timing
methodology is verifiable today; only the dataset numbers come later.

Dependencies: numpy (required); opencv-python (required for flow — `--full` tier).
"""
from __future__ import annotations

import argparse
import time
import numpy as np

MAG_THRESHOLD = 0.2


def _load_frames(video: str | None, n: int, h: int, w: int):
    if video:
        import cv2
        cap = cv2.VideoCapture(video)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (w, h))
            frames.append(f.astype(np.float32))
        cap.release()
        if len(frames) < 2:
            raise RuntimeError("video yielded <2 frames")
        return frames
    # synthetic: a bright blob translating across a static textured background
    rng = np.random.default_rng(0)
    bg = (rng.random((h, w)) * 60).astype(np.float32)
    frames = []
    for t in range(n):
        f = bg.copy()
        cy, cx = h // 2, int(w * 0.15 + t * (w * 0.7 / max(n - 1, 1)))
        yy, xx = np.ogrid[:h, :w]
        f[(yy - cy) ** 2 + (xx - cx) ** 2 < (h // 8) ** 2] = 220
        frames.append(f)
    return frames


def _flow(prev, cur):
    import cv2
    return cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 10, 3, 5, 1.2, 0)


def _separate(frames):
    import cv2
    # motion pass
    for i in range(1, len(frames)):
        fl = _flow(frames[i - 1], frames[i])
        mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        _ = (mag > MAG_THRESHOLD).astype(np.uint8)
    # background pass (recomputes the SAME flow)
    for i in range(1, len(frames)):
        fl = _flow(frames[i - 1], frames[i])
        mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        _ = (mag <= MAG_THRESHOLD).astype(np.uint8)


def _unified(frames):
    import cv2
    for i in range(1, len(frames)):
        fl = _flow(frames[i - 1], frames[i])
        mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        motion = (mag > MAG_THRESHOLD).astype(np.uint8)
        _ = 1 - motion  # background derived for free


def benchmark(frames, repeats: int = 5):
    def timeit(fn):
        fn(frames)  # warmup
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(frames)
        return (time.perf_counter() - t0) / repeats

    t_sep = timeit(_separate)
    t_uni = timeit(_unified)
    print("=" * 60)
    print(f"frames={len(frames)}  repeats={repeats}")
    print(f"separate (2x flow): {t_sep * 1e3:8.2f} ms/clip")
    print(f"unified  (1x flow): {t_uni * 1e3:8.2f} ms/clip")
    print(f"speedup            : {t_sep / t_uni:6.2f}x  "
          f"({(1 - t_uni / t_sep) * 100:.1f}% reduction)")
    return {"separate_ms": t_sep * 1e3, "unified_ms": t_uni * 1e3,
            "speedup": t_sep / t_uni}


def main():
    ap = argparse.ArgumentParser(description="Unified vs separate optical-flow timing")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--h", type=int, default=224)
    ap.add_argument("--w", type=int, default=224)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    try:
        import cv2  # noqa: F401
    except Exception:
        print("opencv-python not installed. Run: bash scripts/setup_env.sh --full")
        return
    frames = _load_frames(args.video, args.n, args.h, args.w)
    benchmark(frames, args.repeats)


if __name__ == "__main__":
    main()
