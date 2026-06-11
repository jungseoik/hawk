"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

# --- Compatibility shim (CERBERUS) ------------------------------------------
# torchvision >= 0.17 removed the public `torchvision.transforms.functional_tensor`
# module, but pytorchvideo 0.1.5 (pulled in via ImageBind) still imports it. The
# functions live on in the private `_functional_tensor`. Alias it so the env built
# by scripts/setup_env.sh (Blackwell torch cu128 + new torchvision) can import.
try:  # pragma: no cover
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
# ---------------------------------------------------------------------------

from omegaconf import OmegaConf

from hawk.common.registry import registry

from hawk.datasets.builders import *
from hawk.models import *
from hawk.processors import *
from hawk.tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
