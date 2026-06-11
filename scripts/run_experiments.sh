#!/usr/bin/env bash
# Run the dataset-free diagnostic experiments (E1, E3, efficiency).
# These validate the metric/loss/efficiency machinery before data arrives.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${ENV_NAME:-cerberus}"
RUN() { conda run -n "${ENV_NAME}" python "$@"; }

echo "########## E1 — CDS disentanglement self-test ##########"
RUN experiments/disentanglement.py --self-test

echo; echo "########## E3 — loss-direction causal demo ##########"
RUN experiments/loss_direction.py

echo; echo "########## E1/E3 — representation viz (synthetic) ##########"
RUN experiments/representation_viz.py --regime separated --method pca --out experiments/out/repviz_separated.png
RUN experiments/representation_viz.py --regime collapsed --method pca --out experiments/out/repviz_collapsed.png

echo; echo "########## E2 — background-swap / BSI self-test ##########"
RUN experiments/bg_critical_benchmark/background_swap.py --self-test

echo; echo "########## Efficiency — unified vs separate optical flow ##########"
RUN experiments/efficiency_benchmark.py || \
  echo "(efficiency needs opencv: bash scripts/setup_env.sh --full)"

echo; echo "All dataset-free experiments finished. Figures in experiments/out/."
