# Reproduce CERBERUS from a fresh clone

> End-to-end, copy-pasteable path from `git clone` to a running Stage-1 training,
> on modern GPUs (incl. NVIDIA **Blackwell**). Every step has a verification check.
> For the data story (why parquet → mp4, the gotchas) see
> [`data_webvid_setup.md`](data_webvid_setup.md); for the model/loss design see
> [`tri-branch-architecture.md`](tri-branch-architecture.md) and
> [`cerberus-research-plan.md`](cerberus-research-plan.md).

```
clone ─▶ ① env ─▶ ② weights ─▶ ③ data (download→extract) ─▶ ④ smoke test ─▶ ⑤ train ─▶ ⑥ eval
```

## Prerequisites
- Linux, NVIDIA GPU + recent driver. Verified on 2× RTX PRO 6000 (Blackwell, sm_120), driver 580, CUDA 12.8.
- `conda` (miniconda/anaconda).
- Disk: ~1.1 TB for the 2M parquet + ~1.2 TB for extracted mp4 (single fast disk recommended; delete parquet after extraction to reclaim).
- A Hugging Face account/token (`hf auth login`) — the model weights repo is public but login avoids rate limits.

---

## ① Environment (Blackwell-compatible)

The upstream `environment.yml` pins `torch==2.0.1+cu117`, which does **not** support Blackwell (sm_120). Use the provided setup script, which builds a `cerberus` conda env with a cu128 PyTorch.

```bash
git clone <this-repo> && cd hawk
bash scripts/setup_env.sh --full      # torch cu128 + full HAWK training stack
conda activate cerberus
pip install -e .                      # register the `hawk` package on the path
```
**Verify**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# expect: 2.x+cu128 True (12, 0)
python -c "import hawk.models.video_llama"   # imports clean (torchvision shim handles pytorchvideo)
```
> `--analysis` instead of `--full` installs only the diagnostic stack (numpy/sklearn/…) for the `experiments/` code, without the heavy training deps.

---

## ② Model weights (LLaMA-2-7B-chat)

Stage-1 initializes the appearance Q-Former from BLIP-2 (auto-downloaded URL) and EVA-ViT-g (`eva_clip_g`, auto-downloaded); the only weight you fetch manually is the base LLM.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned \
  --include "llama-2-7b-chat-hf/*" \
  --local-dir weights/Video-LLaMA-2-7B-Finetuned
```
The configs already point `llama_model` here. **Verify**
```bash
python -c "from transformers import LlamaTokenizer as T; T.from_pretrained('weights/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf', use_fast=False); print('ok')"
```

---

## ③ Data: download (parquet) → extract (mp4 + caption)

The original WebVid was taken down (Shutterstock C&D, 2024-02); we use the
`jxie/webvid_10m` mirror, which embeds the actual **video bytes** in parquet.
Full detail + scale table: [`data_webvid_setup.md`](data_webvid_setup.md).

**Download ~2M clips** (main + part_0 + part_1 ≈ 2.6M, ~1.1 TB) with the
stall/slowdown-resilient watchdog (auto cancel+resume on throttling):
```bash
nohup python scripts/resilient_hf_download.py \
  --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
  --base /data/pia --include "data/*.parquet" --stall 300 --min-mbps 3 \
  > /data/pia/watchdog.log 2>&1 &
# progress: ls /data/pia/webvid_10m*/data/*.parquet | wc -l   (target ~1330)
```
*(For a quick run use only `jxie/webvid_10m` ≈ 200K clips, ~85 GB.)*

**Extract** the parquet "boxes" into individual mp4 files + caption CSVs that the
dataloader reads (byte copy, no re-encode):
```bash
python scripts/build_webvid_split.py --single /data/pia/webvid_extracted
# -> /data/pia/webvid_extracted/{videos/<id>/*.mp4, annotations/*.csv}
```
Then point the training config at it (edit `configs/train_configs/stage1_pretrain.yaml`):
```yaml
datasets:
  webvid:
    build_info:
      anno_dir:   /data/pia/webvid_extracted/annotations/
      videos_dir: /data/pia/webvid_extracted/videos/
```
**Verify** one sample loads (3 streams + 3 captions):
```bash
python - <<'PY'
from hawk.processors.video_processor import AlproVideoTrainProcessor
from hawk.processors.blip_processors import BlipCaptionProcessor
from hawk.datasets.datasets.webvid_datasets import WebvidDataset
ds = WebvidDataset(AlproVideoTrainProcessor(image_size=224, n_frms=32), BlipCaptionProcessor(),
                   vis_root="/data/pia/webvid_extracted/videos", ann_root="/data/pia/webvid_extracted/annotations")
s = ds[0]; print({k: getattr(v,'shape',v) for k,v in s.items() if k.startswith('image')}); print(s['text_input_motion'], '|', s['text_input_background'])
PY
```

**Reclaim space — delete the parquet (after extraction is verified).** The parquet
files were only the download container; the extracted mp4 + CSV are now the
dataset, so the ~1.1 TB of parquet is fully redundant. Delete it **only after**
confirming extraction finished (`videos/` page_dir count == shard count and the
verify above passes). Skip this if you plan to re-extract (e.g. retune the flow
threshold).
```bash
ls /data/pia/webvid_extracted/videos | wc -l     # == number of downloaded shards?
rm -rf /data/pia/webvid_10m*/data                # reclaim ~1.1 TB
```

---

## ④ Smoke test (no full training)

Confirms the whole model builds + forward/backward run on your GPU before a long run.
```bash
conda run -n cerberus python scripts/smoke_test.py --frames 8         # build + dummy vision forward
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29561 \
  train.py --cfg-path configs/train_configs/stage1_smoke.yaml         # 5 real training iters
```
**Expect**: model `10.18B` params / `170M` trainable; 5 iters with decreasing
`totalloss`; a `checkpoint_0.pth` written under the config's `output_dir`.
(Reference: ~1.2 s/iter, ~62 GB VRAM at batch 1, 32 frames × 3 streams.)

---

## ⑤ Train (2-stage)

```bash
# Stage 1 — pretrain on WebVid (5 losses, per-stream supervision)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=10000 \
  train.py --cfg-path configs/train_configs/stage1_pretrain.yaml

# Stage 2 — finetune on the anomaly dataset (set `ckpt:` to the Stage-1 output)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12001 \
  train.py --cfg-path configs/train_configs/stage2_finetune.yaml
```
- Total steps = `max_epoch × iters_per_epoch` (edit to your compute budget).
- TensorBoard: `Loss/total`, `Loss/ori`, `Loss/motion`, `Loss/background`,
  `Loss/middle` (app↔motion), `Loss/middle_bg` (motion↔background).
- Stage-2 anomaly data (HAWK 7-dataset): see README → Dataset Preparation.

---

## ⑥ Evaluate / demo
```bash
python app.py --cfg-path configs/eval_configs/eval.yaml --model_type llama_v2 --gpu-id 0
```
Diagnostic experiments (CDS / BSI / loss-direction, no dataset needed):
```bash
bash scripts/run_experiments.sh        # see experiments/README.md
```

---

## Troubleshooting (verified gotchas)

| 증상 | 원인 / 해결 |
|---|---|
| `torch ... no kernel image ... sm_120` | cu117 torch on Blackwell → use `scripts/setup_env.sh` (cu128). |
| `No module named 'hawk'` | run `pip install -e .`, or run scripts from the repo root. |
| `torchvision.transforms.functional_tensor` ImportError | handled by a shim in `hawk/__init__.py` (pytorchvideo 0.1.5 vs new torchvision). |
| cv2 `_ARRAY_API not found` | opencv vs numpy 2.x → `setup_env.sh` installs `opencv-python>=4.10`. |
| HF download crawls at ~0.1 MB/s | resumed-connection throttling → `resilient_hf_download.py` auto cancels + re-fetches fresh. |
| `__getitem__` hangs / infinite loop | page_dir parsed as int (numeric) → the extractor prefixes `m/a/b` (non-numeric) on purpose. |

## Hardware notes
- Verified: 2× RTX PRO 6000 (Blackwell, 96 GB). batch 1 / 32 frames / 3 streams ≈ 62 GB VRAM.
- Original HAWK used 4× A6000 (48 GB). On fewer/smaller GPUs, reduce `iters_per_epoch`/`max_epoch` or use gradient accumulation.
