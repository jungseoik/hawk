# Improvements: Optical Flow Deduplication & Loss Direction Fix

> 📌 본 문서는 **신규 연구 CERBERUS**(HAWK의 Tri-Branch 확장)에 적용된 코드 개선 기록입니다. 베이스 HAWK와의 구분은 [`../CLAUDE.md`](../CLAUDE.md)를 참고하세요.

## Overview

Two architectural improvements were applied to the CERBERUS tri-branch framework to address computational inefficiency and a training objective misalignment.

---

## 1. Optical Flow Duplicate Computation Removal

### Problem

`load_video_motion()` and `load_video_background()` each independently computed Farneback optical flow on the same frame pairs. Since both functions read the full video and run `cv2.calcOpticalFlowFarneback()` per frame, the expensive optical flow computation was executed **twice** for every video sample.

**Before (2x optical flow):**
```
load_video_motion()   → read video → compute optical flow → apply motion mask
load_video_background() → read video → compute optical flow → apply inverse mask
```

### Solution

Introduced `compute_motion_and_background()` which computes optical flow **once** and produces both motion and background frames from the same flow field. The background mask is derived as the bitwise complement of the motion mask (`1 - motion_mask_3ch`), which is both mathematically equivalent and avoids a redundant threshold comparison.

A new unified loader `load_video_motion_and_background()` reads the video once and returns both tensors.

**After (1x optical flow):**
```
load_video_motion_and_background() → read video → compute optical flow once → split into motion + background
```

### Files Changed

| File | Change |
|------|--------|
| `hawk/processors/video_processor.py` | Added `compute_motion_and_background()`, `load_video_motion_and_background()`; kept original functions as thin wrappers for backward compatibility |
| `hawk/processors/video_processor.py` | Updated `AlproVideoTrainProcessor.__call__()` and `AlproVideoEvalProcessor.__call__()` to use unified loader |
| `hawk/datasets/datasets/video_instruct_dataset.py` | Updated `__getitem__()` to use `load_video_motion_and_background()` |

### Impact

- ~50% reduction in optical flow computation time per sample
- ~33% reduction in video I/O (2 reads instead of 3 for motion+background)
- No change to model outputs or training behavior

---

## 2. Motion-Background Dissimilarity Loss Direction Fix

### Problem

The loss `L_middle(motion, background)` was implemented as:

```python
mse_loss_bg = F.cosine_similarity(motion, background)
mse_loss_bg = 1 - mse_loss_bg  # BUG: encourages similarity
```

This formulation yields `loss = 1 - cos_sim`. Minimizing this loss **maximizes** cosine similarity, pushing motion and background representations to be **similar**. This contradicts the design intent: motion and background are complementary (inverse masks), so their representations should be **dissimilar**.

The code comment correctly stated the intent ("background should be dissimilar from motion"), but the implementation achieved the opposite.

### Solution

Changed to:

```python
mse_loss_bg = F.cosine_similarity(motion, background)
mse_loss_bg = (1 + mse_loss_bg) / 2  # FIX: encourages dissimilarity
```

This formulation:
- Maps cosine similarity from [-1, 1] to [0, 1]
- Minimizing loss pushes `cos_sim` toward -1 (maximum dissimilarity)
- Loss = 0 when representations are perfectly anti-correlated
- Loss = 1 when representations are identical

### Comparison

| cos_sim | Old loss (1 - cos) | New loss (1 + cos)/2 | Gradient direction |
|---------|--------------------|-----------------------|-------------------|
| 1.0 (identical) | 0.0 (no penalty) | 1.0 (max penalty) | Old: no push; **New: push apart** |
| 0.0 (orthogonal) | 1.0 | 0.5 | Old: push together; **New: push apart** |
| -1.0 (opposite) | 2.0 | 0.0 (no penalty) | Old: strong push together; **New: satisfied** |

### Files Changed

| File | Change |
|------|--------|
| `hawk/tasks/base_task.py` | Fixed `mse_loss_bg` computation from `1 - cos_sim` to `(1 + cos_sim) / 2` |

### Impact

- Motion and background branches now correctly learn complementary (non-overlapping) representations
- Expected improvement in anomaly detection quality, as each branch focuses on its designated domain without information redundancy
- The appearance-motion similarity loss (`L_middle(app, motion)`) remains unchanged (`1 - cos_sim`), correctly encouraging those two branches to share mutual information
