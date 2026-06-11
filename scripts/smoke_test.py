"""
Model build + forward smoke test (no dataset needed).
=====================================================
De-risks full retraining BEFORE data arrives: builds the full tri-stream model
from a training config (downloads EVA-ViT / BLIP-2 Q-Former / bert-base-uncased,
loads the local LLaMA-2-7B) and runs one dummy vision forward through each of the
appearance / motion / background streams on the GPU.

Validates the Blackwell env (torch cu128, transformers 4.28, new torchvision)
can construct and run the model end-to-end. It does NOT train and does NOT need
WebVid / anomaly datasets.

Usage:
    conda run -n cerberus python scripts/smoke_test.py \
        --cfg configs/train_configs/stage1_pretrain.yaml --frames 8
"""
import argparse, sys, traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_configs/stage1_pretrain.yaml")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    import torch
    from hawk.common.config import Config
    from hawk.common.registry import registry
    import hawk.models  # noqa: F401  register arch

    class _A:
        def __init__(self, p):
            self.cfg_path = p
            self.options = None

    print(f"[smoke] loading config: {args.cfg}")
    cfg = Config(_A(args.cfg))
    model_cfg = cfg.model_cfg
    print(f"[smoke] arch={model_cfg.arch}  llama_model={model_cfg.llama_model}")

    print("[smoke] building model (downloads EVA-ViT / Q-Former / bert on first run, loads LLaMA)...")
    model = registry.get_model_class(model_cfg.arch).from_config(model_cfg)
    model = model.eval().cuda()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model built. params: total={n_total/1e9:.2f}B  trainable={n_train/1e6:.1f}M")

    B, C, T, H, W = 1, 3, args.frames, 224, 224
    dummy = torch.randn(B, C, T, H, W, dtype=torch.float16, device="cuda")
    print(f"[smoke] dummy video: {tuple(dummy.shape)}")

    for label, kw in [("appearance", {}), ("motion", {"motion": True}),
                      ("background", {"background": True})]:
        with torch.no_grad():
            emb, atts, middle = model.encode_videoQformer_visual(dummy, **kw)
        print(f"[smoke]   {label:11s}: llama_embed={tuple(emb.shape)}  middle={tuple(middle.shape)}")

    print("[smoke] PASS — full tri-stream model builds and runs a forward on Blackwell.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
