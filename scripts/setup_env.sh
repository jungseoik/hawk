#!/usr/bin/env bash
# =============================================================================
# CERBERUS — Environment setup (Blackwell-compatible)
# =============================================================================
# The original HAWK environment.yml pins torch==2.0.1+cu117, which does NOT
# support NVIDIA Blackwell GPUs (sm_120, e.g. RTX PRO 6000). Blackwell requires
# CUDA 12.8+ runtime and torch >= 2.7 (cu128 wheels). This script builds a
# Blackwell-compatible conda env named `cerberus`.
#
# Two install tiers:
#   --analysis  : torch + numpy/scipy/scikit-learn/matplotlib/umap (enough to run
#                 the diagnostic experiments E1-E4 analysis code; CPU/GPU both OK)
#   --full      : analysis tier + the full HAWK training stack (transformers,
#                 opencv, spacy, decord, etc.) needed for Stage1/Stage2 training
#
# Usage:
#   bash scripts/setup_env.sh --analysis        # fast, for E1-E4 analysis code
#   bash scripts/setup_env.sh --full            # everything (heavier)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-cerberus}"
PY_VERSION="3.10"
TIER="${1:---analysis}"

echo "[setup_env] conda: $(which conda)"
echo "[setup_env] target env: ${ENV_NAME}  (python ${PY_VERSION})  tier: ${TIER}"

# ---------------------------------------------------------------------------
# 1. Create env if missing
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup_env] env '${ENV_NAME}' already exists — reusing."
else
  echo "[setup_env] creating env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

# Resolve env python/pip without needing `conda activate` in non-interactive shell
ENV_PY="$(conda run -n "${ENV_NAME}" which python)"
echo "[setup_env] env python: ${ENV_PY}"

run_pip() { conda run -n "${ENV_NAME}" python -m pip "$@"; }

run_pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. Blackwell-compatible PyTorch (cu128)
# ---------------------------------------------------------------------------
echo "[setup_env] installing torch (cu128, Blackwell-compatible)..."
run_pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

# ---------------------------------------------------------------------------
# 3. Analysis tier — diagnostic experiments E1-E4
# ---------------------------------------------------------------------------
echo "[setup_env] installing analysis stack..."
run_pip install \
  numpy scipy scikit-learn matplotlib pandas \
  umap-learn seaborn tqdm pyyaml

# ---------------------------------------------------------------------------
# 4. Full tier — HAWK training stack (optional)
# ---------------------------------------------------------------------------
if [ "${TIER}" = "--full" ]; then
  echo "[setup_env] installing full HAWK training stack (pins matched to HAWK environment.yml)..."
  # NOTE: torch stays at the Blackwell-compatible cu128 build installed above.
  # Only the non-torch deps are pinned to HAWK's versions for code compatibility.
  # transformers 4.28.0 is what hawk/models/{video_llama,Qformer,modeling_llama}.py
  # were written against — do not bump without checking those imports.

  # PyAV + ffmpeg via conda-forge: the av==10 sdist needs system ffmpeg dev headers
  # (libavformat-dev ...) to compile, which require sudo. conda-forge ships av with
  # bundled ffmpeg libraries, so no system packages / root are needed.
  echo "[setup_env] installing av + ffmpeg via conda-forge (bundled libs, no sudo)..."
  conda install -y -n "${ENV_NAME}" -c conda-forge av ffmpeg

  run_pip install \
    "transformers==4.28.0" "tokenizers==0.13.3" \
    "sentencepiece==0.1.97" "accelerate==0.23.0" "peft==0.5.0" \
    "opencv-python==4.8.1.78" "decord==0.6.0" \
    imageio imageio-ffmpeg pytorchvideo \
    "timm==0.9.7" "einops==0.8.1" "omegaconf==2.3.0" "iopath==0.1.10" \
    "spacy==3.8.4" "webdataset==0.2.57" "ftfy==6.3.1" "regex==2023.10.3" \
    tensorboard "gradio==5.17.1"
  echo "[setup_env] downloading spaCy en_core_web_sm..."
  conda run -n "${ENV_NAME}" python -m spacy download en_core_web_sm || \
    echo "[setup_env] WARN: spaCy model download failed (retry online later)."
fi

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
echo "[setup_env] verifying torch / CUDA / Blackwell capability..."
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {name}  sm_{cap[0]}{cap[1]}")
    archs = torch.cuda.get_arch_list()
    print("compiled arch list:", archs)
    print("sm_120 supported:", any("sm_120" in a or "12.0" in a for a in archs))
PY

echo "[setup_env] DONE. Activate with:  conda activate ${ENV_NAME}"
