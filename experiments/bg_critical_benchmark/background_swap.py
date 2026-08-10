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


def corrected_bsi(resp_safe, resp_danger, resp_safe_alt, resp_repeat=None, embed_fn=None):
    """귀무 조건을 감산한 BSI.

    왜 필요한가. 원래의 `bsi()` 는 배경을 바꿨을 때 응답이 얼마나 달라지는가만 잰다.
    그런데 응답이 달라지는 이유는 두 가지이고, 원 지표는 둘을 구분하지 못한다.

      (a) 모델이 배경의 **의미**를 인과적으로 사용한다  ← 논문이 주장하려는 것
      (b) 모델이 **어떤** 픽셀 변화에도 불안정하다        ← 무해한 대안 설명

    같은 데이터가 두 해석에 동일하게 부합하므로, 귀무 조건 없이 얻은 BSI 는 인과 주장을
    지지하지 않는다. `ΔBSI`(정적 스트림 유/무의 차분)도 이를 상쇄하지 못한다 — 두 모델이
    합성 아티팩트에 서로 다르게 민감할 수 있고, 그 차이가 그대로 ΔBSI 로 나타난다.

    따라서 세 값을 함께 재고 감산한다.

      resp_safe / resp_danger    안전 배경 vs 위험 배경          → 신호
      resp_safe_alt              **의미가 동등한 다른 안전 배경**  → 귀무 (BSI ≈ 0 기대)
      resp_repeat                동일 입력 재생성 (선택)          → 생성 자체의 변동

    반환하는 `corrected` 가 보고 대상이며, 이것이 0 이하면 "배경 의미에 반응했다"고 말할 수
    없다. `null` 이 크면 지표가 의미가 아니라 섭동 크기를 재고 있다는 뜻이므로, 그 경우
    합성 품질(경계 아티팩트·조도 불일치)을 먼저 점검해야 한다.
    """
    signal = bsi(resp_safe, resp_danger, embed_fn)
    null = bsi(resp_safe, resp_safe_alt, embed_fn)
    out = {"signal": signal, "null": null, "corrected": signal - null}
    if resp_repeat is not None:
        # 동일 입력을 두 번 생성했을 때의 차이. greedy 디코딩이면 0 이어야 한다.
        # 0 이 아니면 표본 추출이 켜져 있다는 뜻이고, 그 변동은 BSI 에 양의 편향으로
        # 들어간다 (scripts/evaluate.py 는 do_sample=False 로 고정한다).
        out["self_consistency"] = bsi(resp_safe, resp_repeat, embed_fn)
    return out


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
