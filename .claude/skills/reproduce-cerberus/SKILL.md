---
name: reproduce-cerberus
description: >-
  Reproduce the CERBERUS (a.k.a. HAWK++) video-anomaly training pipeline end-to-end
  from this repo — build the Blackwell-compatible conda env, fetch the LLaMA-2 base
  weights, download + extract the WebVid dataset (parquet → mp4), run a smoke test,
  then launch Stage-1/Stage-2 training with checkpointing, stop/resume, and TensorBoard
  logging. Use whenever asked to reproduce, set up from scratch, install dependencies for,
  download data for, smoke-test, or train/run CERBERUS/HAWK in this repository.
---

# Reproduce CERBERUS end-to-end

Authoritative long-form guide: **[`docs/reproduce.md`](../../../docs/reproduce.md)** and
**[`docs/data_webvid_setup.md`](../../../docs/data_webvid_setup.md)**. This skill is the
agent-actionable runbook — follow the steps **in order**, run the verify check after each,
and stop/report if a verify fails.

## What "reproduce" means here
`git clone` gives you all code / configs / scripts / docs / paper (incl. the HAWK PDF), but
**not** the dataset or model weights (excluded by design — copyright + ~1 TB size). Reproduction =
clone + run these scripts to fetch weights & data, then train. It is not instant.

## Machine assumptions — ADAPT FIRST
The scripts default to this dev box; change these if the target machine differs:
- **Env name** `cerberus` (conda). GPUs: 2× (Blackwell sm_120). `--nproc_per_node` = #GPUs.
- **Data/weights dir**: big writable disk. Here `/data/pia` (3.4 TB). Verify a writable path
  with ≥ ~1.3 TB free (`df -h`); if not `/data/pia`, replace that prefix in the commands and
  in `configs/train_configs/stage1_pretrain.yaml` (`anno_dir`/`videos_dir`).
- **HF auth**: `hf auth status` must show the correct account (needs `repo` scope). If wrong,
  `gh auth login` / `hf auth login` as the right user.

---

## ① Environment + dependencies
```bash
bash scripts/setup_env.sh --full        # conda env `cerberus`: torch cu128 (Blackwell), full stack,
                                        # pyarrow, hf_transfer, spaCy model, and `pip install -e .`
conda activate cerberus
```
**Verify**
```bash
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability(0))"  # 2.x+cu128 True (12,0)
python -c "import hawk.models.video_llama, pyarrow, cv2, spacy; spacy.load('en_core_web_sm'); print('deps ok')"
```

## ② LLaMA-2-7B-chat weights (~13 GB)
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned \
  --include "llama-2-7b-chat-hf/*" --local-dir weights/Video-LLaMA-2-7B-Finetuned
```
Configs already point `llama_model` here. EVA-ViT + BLIP-2 Q-Former + bert-base auto-download at build.
**Verify**: `python -c "from transformers import LlamaTokenizer as T;T.from_pretrained('weights/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf',use_fast=False);print('ok')"`

## ③ Data: download (parquet) → extract (mp4 + captions)
Resilient downloader auto-recovers stalls/slowdowns (kills + resumes fresh). ~2M-clip target
= main + part_0 + part_1 (~1.1 TB, hours). For a quick run use only `jxie/webvid_10m` (~85 GB, 200K clips).
```bash
nohup python scripts/resilient_hf_download.py \
  --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
  --base /data/pia --include "data/*.parquet" --stall 300 --min-mbps 3 \
  > /data/pia/watchdog.log 2>&1 &
# progress: ls /data/pia/webvid_10m*/data/*.parquet | wc -l   ;  tail /data/pia/watchdog.log
```
When download finishes, extract to individual mp4 (single disk; byte-copy, no re-encode):
```bash
python scripts/build_webvid_split.py --single /data/pia/webvid_extracted
```
Point `configs/train_configs/stage1_pretrain.yaml` → `anno_dir=/data/pia/webvid_extracted/annotations/`,
`videos_dir=/data/pia/webvid_extracted/videos/` (already set for this box).
**Verify** one sample (3 streams + 3 captions):
```bash
python - <<'PY'
from hawk.processors.video_processor import AlproVideoTrainProcessor
from hawk.processors.blip_processors import BlipCaptionProcessor
from hawk.datasets.datasets.webvid_datasets import WebvidDataset
ds=WebvidDataset(AlproVideoTrainProcessor(image_size=224,n_frms=32),BlipCaptionProcessor(),
   vis_root="/data/pia/webvid_extracted/videos",ann_root="/data/pia/webvid_extracted/annotations")
s=ds[0]; print(len(ds), s['image'].shape, '|', s['text_input_motion'], '|', s['text_input_background'])
PY
```
Optional space reclaim (only after verify): `rm -rf /data/pia/webvid_10m*/data`  (~1.1 TB)

## ④ Smoke test (build + a few real iters, no full run)
```bash
conda run -n cerberus python scripts/smoke_test.py --frames 8
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29561 \
  train.py --cfg-path configs/train_configs/stage1_smoke.yaml
```
**Expect**: model 10.18B params / 170M trainable; 5 iters, decreasing `totalloss`; a `checkpoint_0.pth`.

## ⑤ Train (2-stage)
```bash
# Stage 1 — pretrain on WebVid
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=10000 \
  train.py --cfg-path configs/train_configs/stage1_pretrain.yaml
# Stage 2 — finetune on anomaly data (set stage2 `ckpt:` to the Stage-1 output checkpoint)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12001 \
  train.py --cfg-path configs/train_configs/stage2_finetune.yaml
```
- Total steps = `max_epoch × iters_per_epoch` (set for compute budget). Measured throughput on 2×
  Blackwell ≈ 3 clips/s (compute-bound; larger batch helps little). ~6 days for HAWK-equiv 1.6M
  samples, ~10 days for one pass over 2.63M. batch 4 ≈ 75 GB VRAM (safe max).
- **Logs**: per-iter CLI losses; per-job TensorBoard at `<output_dir>/<job_id>/tensorboard`.
- **Checkpoints**: every epoch → `<output_dir>/<job_id>/checkpoint_<N>.pth` (weights+optimizer+scaler+epoch).
- **Stop/resume (바통터치)**: set `run.resume_ckpt_path: <...>/checkpoint_<N>.pth` in the config and
  relaunch → resumes at epoch N+1 with optimizer/LR/scaler restored ("Resume checkpoint from ...").
  Quick sanity config: `configs/train_configs/stage1_validate.yaml` (3 ep × 15 it, ~1 min).

## ⑥ Evaluate / diagnostics
```bash
python app.py --cfg-path configs/eval_configs/eval.yaml --model_type llama_v2 --gpu-id 0
bash scripts/run_experiments.sh        # CDS / BSI / loss-direction diagnostics (no dataset needed)
```

## Troubleshooting (verified gotchas)
| symptom | fix |
|---|---|
| `no kernel image ... sm_120` | cu117 torch on Blackwell → use `setup_env.sh` (cu128). |
| `No module named 'hawk'` | `pip install -e .` (setup_env.sh does this) or run from repo root. |
| `torchvision.transforms.functional_tensor` ImportError | shim in `hawk/__init__.py` (pytorchvideo vs new torchvision). |
| cv2 `_ARRAY_API not found` | numpy 2 vs old opencv → `opencv-python>=4.10` (in setup_env.sh). |
| HF download crawls ~0.1 MB/s | `resilient_hf_download.py` auto cancels + re-fetches fresh. |
| `__getitem__` hangs | page_dir parsed as int → extractor prefixes `m/a/b` (non-numeric); do not remove. |
| resume `torch.load ... unexpected 'strict'` | fixed in `runner_base._load_checkpoint` (already patched). |

## Report format
When done, report: which steps passed their verify, dataset scale fetched (shards/clips), any error hit +
resolution, and where checkpoints/TensorBoard landed. Never fabricate results — only report what ran.
