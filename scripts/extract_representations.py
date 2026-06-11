"""
Dump middle representations (z_a, z_m, z_b) for E1/E3 analysis.
===============================================================
The model computes three compact "middle" representations inside
`hawk/models/video_llama.py::encode_videoQformer_visual` (returned as the
`middle` tensor) for the appearance, motion and background branches
(`middle_result`, `middle_result_motion`, `middle_result_background` in
`forward`). This script runs the model over a list of test clips and saves those
vectors to an .npz consumed by:
    experiments/disentanglement.py --reps <out.npz>
    experiments/representation_viz.py --reps <out.npz>

Because it needs the trained checkpoint + videos (integrated LAST), the real
path is guarded. Use --dry-run to emit a synthetic npz today and validate the
whole downstream analysis pipeline.

Real-run outline (wire when weights/data are ready):
    cfg   = Config(...)              # from configs/eval_configs/eval.yaml
    model = registry.get_model_class(cfg.arch).from_config(cfg).eval().cuda()
    for clip in clips:
        x_a = load_video(clip); x_m, x_b = load_video_motion_and_background(clip)
        _, _, z_a = model.encode_videoQformer_visual(x_a)
        _, _, z_m = model.encode_videoQformer_visual(x_m, motion=True)
        _, _, z_b = model.encode_videoQformer_visual(x_b, background=True)
        collect z_a.mean(1), z_m.mean(1), z_b.mean(1)   # pool query tokens
"""
from __future__ import annotations

import argparse
import numpy as np


def dry_run(out: str, n: int = 256, d: int = 256, seed: int = 0):
    """Emit a synthetic reps.npz so the analysis pipeline is testable now."""
    rng = np.random.default_rng(seed)
    fm = rng.standard_normal((n, d))
    fb = rng.standard_normal((n, d))
    W = rng.standard_normal((2 * d, d))
    z_a = np.concatenate([fm, fb], 1) @ W + 0.1 * rng.standard_normal((n, d))
    np.savez(out, z_a=z_a.astype(np.float32),
             z_m=fm.astype(np.float32), z_b=fb.astype(np.float32))
    print(f"[dry-run] wrote synthetic reps -> {out}  (z_a,z_m,z_b: {n}x{d})")


def real_run(cfg_path, ckpt, clips_file, out, pool="mean"):  # pragma: no cover
    import torch
    from hawk.common.config import Config
    from hawk.common.registry import registry
    import hawk.models  # noqa: F401  (register arch)
    from hawk.processors.video_processor import (
        load_video, load_video_motion_and_background)

    class _A:  # minimal args shim for Config
        def __init__(self, p): self.cfg_path = p; self.options = None
    cfg = Config(_A(cfg_path))
    if ckpt:
        cfg.model_cfg.ckpt = ckpt
    model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
    model = model.eval().to("cuda")

    with open(clips_file) as f:
        clips = [ln.strip() for ln in f if ln.strip()]

    Za, Zm, Zb = [], [], []
    for clip in clips:
        x_a = load_video(clip).unsqueeze(0).to("cuda")
        x_m, x_b, _ = load_video_motion_and_background(clip)
        x_m = x_m.unsqueeze(0).to("cuda"); x_b = x_b.unsqueeze(0).to("cuda")
        with torch.no_grad():
            _, _, z_a = model.encode_videoQformer_visual(x_a)
            _, _, z_m = model.encode_videoQformer_visual(x_m, motion=True)
            _, _, z_b = model.encode_videoQformer_visual(x_b, background=True)
        red = (lambda t: t.mean(1)) if pool == "mean" else (lambda t: t.flatten(1))
        Za.append(red(z_a).cpu().numpy()); Zm.append(red(z_m).cpu().numpy())
        Zb.append(red(z_b).cpu().numpy())
    np.savez(out, z_a=np.concatenate(Za), z_m=np.concatenate(Zm), z_b=np.concatenate(Zb))
    print(f"[real] wrote {len(clips)} clips of reps -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Dump z_a/z_m/z_b middle reps")
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--clips", default=None, help="text file: one video path per line")
    ap.add_argument("--out", default="experiments/out/reps.npz")
    ap.add_argument("--pool", choices=["mean", "flatten"], default="mean")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit synthetic reps (no model/data needed)")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.dry_run or not args.clips:
        dry_run(args.out)
    else:
        real_run(args.cfg, args.ckpt, args.clips, args.out, args.pool)


if __name__ == "__main__":
    main()
