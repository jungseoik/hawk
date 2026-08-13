#!/usr/bin/env python
"""자기 운동(ego-motion) 보정이 마스크 비율을 실제로 얼마나 바꾸는지 측정한다.

왜 먼저 재는가
--------------
로드맵 ④(ego-motion 보정 arm)는 배경이 원인인 클립의 88%가 있는 DoTA 에서 분할이
퇴화하는 문제를 겨냥한다. 대시캠 영상은 카메라가 함께 움직여 화면 전체가 흐르므로,
플로우 크기로 자르면 정적 스트림이 비어 버린다(DoTA 마스크 비율 0.449 — 고정 카메라
도메인의 0.04~0.16 과 대비된다).

그러나 보정이 실제로 마스크를 되돌리는지는 **학습 없이 확인할 수 있다.** 효과가
미미하면 2일을 쓸 이유가 없고, 크면 ④ 를 우선순위로 올릴 근거가 된다. 학습 경로 코드는
절제 실험이 도는 동안 동결이므로, 이 스크립트는 기존 함수를 건드리지 않고 같은 계산을
독립적으로 수행한다.

보정 방법 — 그리고 처음에 틀렸던 방법
------------------------------------
**올바른 방법(warp-then-flow).** 두 프레임 사이의 전역 변환을 특징점 대응으로 추정하고
(RANSAC 으로 전경 대응을 배제), 앞 프레임을 그 변환으로 **정렬한 뒤** 플로우를 다시
계산한다. 정렬이 성공하면 배경의 플로우는 0 에 가까워지고 독립 운동만 남는다.

    H     = estimateAffinePartial2D(pts_{t-1}, pts_t, RANSAC)
    warped= warpAffine(frame_{t-1}, H)
    flow  = Farneback(warped, frame_t)
    mask  = |flow| > τ

**처음에 시도했다가 틀린 방법(flow 잔차).** 플로우 장에 어파인 모델을 적합하고 그
성분을 빼는 방식은 이 목적에 쓸 수 없다. 실측에서 DoTA 마스크 비율이 0.383 → 0.615 로
**오히려 올라갔다.** 이유는 명확하다 — DoTA 에서도 화소의 62% 는 이미 임계값 아래인데,
전체 플로우 장에 적합된 모델은 그 정지 영역에도 상당한 운동을 예측한다. 그 예측을 빼면
차이가 잔차로 남아 **원래 정적이던 영역이 새로 마스크에 들어간다.** 즉 이 방식은 자기
운동을 제거하는 대신 정적 영역에 인공적인 운동을 주입한다. 기록으로 남긴다 — 같은
착오를 반복하지 않기 위함이다.

사용
----
    $CERBERUS_PY scripts/measure_ego_motion_effect.py --n 60
    $CERBERUS_PY scripts/measure_ego_motion_effect.py --n 40 --datasets DoTA UCF_Crime
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
ANNO = f"{ROOT}/hawk_anomaly/Annotation/All_Mix/all_videos_all.local.json"
VIDEOS = f"{ROOT}/hawk_anomaly/Videos"
MAG_THRESHOLD = 0.2          # video_processor.py 와 동일해야 한다
N_FRAMES = 8                 # 클립당 표본 프레임 (전수 측정이 아니라 비교가 목적)


def align_prev_frame(prev_gray, cur_gray):
    """앞 프레임을 현 프레임에 전역 정렬한다. 실패하면 None.

    특징점 대응 + RANSAC 을 쓰는 이유는 전경이 추정을 끌어가지 않게 하기 위함이다.
    화면의 큰 부분을 차지하는 움직이는 객체가 있어도, RANSAC 은 다수를 이루는 배경
    대응을 inlier 로 골라낸다.
    """
    import cv2
    pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=600, qualityLevel=0.01,
                                       minDistance=8, blockSize=7)
    if pts_prev is None or len(pts_prev) < 20:
        return None, None
    pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts_prev, None)
    if pts_cur is None:
        return None, None
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return None, None
    a, inl = cv2.estimateAffinePartial2D(pts_prev[ok], pts_cur[ok],
                                         method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if a is None or inl is None or int(inl.sum()) < 15:
        return None, None
    h, w = prev_gray.shape
    return cv2.warpAffine(prev_gray, a, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE), a


def probe(rec):
    """한 클립에서 보정 전/후 마스크 비율을 잰다."""
    import cv2
    import decord
    n_align_fail = [0]

    def flow_ratio(prev, cur):
        f = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return float((np.linalg.norm(f, axis=-1) > MAG_THRESHOLD).mean())
    path = os.path.join(VIDEOS, rec["video"])
    try:
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        if n < 3:
            return None
        idx = [int(n * (i + 0.5) / N_FRAMES) for i in range(N_FRAMES)]
        idx = [min(max(i, 1), n - 1) for i in idx]

        raw, corr, ident = [], [], []
        for i in idx:
            a = cv2.cvtColor(vr[i - 1].asnumpy(), cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_RGB2GRAY)
            r0 = flow_ratio(a, b)
            raw.append(r0)

            warped, M = align_prev_frame(a, b)
            if warped is None:
                corr.append(r0)
                ident.append(r0)
                n_align_fail[0] += 1
            else:
                corr.append(flow_ratio(warped, b))
                # ── 통제: 두 프레임을 **같은 변환**으로 warp ────────────────────
                # 상대 운동은 보존되고 보간만 양쪽에 동일하게 걸리므로, 원본 대비
                # 증가분은 전부 보간 아티팩트다. 정렬 조건에서 이 값을 빼면
                # 보간을 제외한 순효과가 남는다.
                #
                # 항등 행렬 통제는 쓰지 않는다 — 정확한 항등은 정수 화소 대응이라
                # 보간을 아예 건너뛰므로 아티팩트가 0 으로 나오고 통제가 되지 않는다
                # (실제로 그렇게 측정되어 잘못된 안심을 준 적이 있다).
                h, w = a.shape
                bw = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                ident.append(flow_ratio(warped, bw))

        return {"video": rec["video"], "dataset": rec["video"].split("/")[0],
                "mask_raw": float(np.mean(raw)), "mask_corrected": float(np.mean(corr)),
                "mask_bothwarp": float(np.mean(ident)),
                "align_failed_frames": n_align_fail[0], "n_frames": len(raw)}
    except Exception as exc:
        return {"video": rec["video"], "error": str(exc)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="데이터셋당 표본 클립 수")
    ap.add_argument("--datasets", nargs="*",
                    default=["DoTA", "UCF_Crime", "UBnormal", "ShanghaiTech"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", default=f"{ROOT}/hawk/experiments/bg_critical_benchmark/ego_motion_effect.json")
    args = ap.parse_args()

    with open(ANNO) as f:
        records = json.load(f)
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["video"].split("/")[0]].append(r)

    rng = random.Random(args.seed)
    targets = []
    for ds in args.datasets:
        pool = by_ds.get(ds, [])
        if pool:
            targets += rng.sample(pool, min(args.n, len(pool)))
    print(f"{len(targets)} 클립 측정 (데이터셋 {args.datasets}, workers={args.workers})")

    results, failed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(probe, r) for r in targets]
        for i, f in enumerate(as_completed(futs), 1):
            out = f.result()
            if out is None or "error" in (out or {}):
                failed += 1
            else:
                results.append(out)
            if i % 50 == 0:
                print(f"  {i}/{len(targets)} (실패 {failed})")

    print(f"\n완료 {len(results)}, 실패 {failed}\n")
    print(f"{'데이터셋':<14}{'n':>4}{'보정 전':>9}{'양쪽warp':>10}{'보정 후':>9}"
          f"{'아티팩트':>10}{'순효과':>10}{'95% CI':>20}")
    summary = {}
    for ds in args.datasets:
        vals = [r for r in results if r["dataset"] == ds]
        if not vals:
            continue
        a = np.array([v["mask_raw"] for v in vals])
        i0 = np.array([v["mask_bothwarp"] for v in vals])
        b = np.array([v["mask_corrected"] for v in vals])
        # 순효과 = (보정 후 − 항등). 항등이 아티팩트의 기준선이므로 이를 감산한다.
        net = b - i0
        art = i0 - a
        se = float(net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 else float("nan")
        lo, hi = float(net.mean() - 1.96 * se), float(net.mean() + 1.96 * se)
        summary[ds] = {"n": len(vals), "raw": float(a.mean()), "bothwarp": float(i0.mean()),
                       "corrected": float(b.mean()), "artifact": float(art.mean()),
                       "net_effect": float(net.mean()), "net_se": se,
                       "net_ci95": [lo, hi]}
        print(f"{ds:<14}{len(vals):>4}{a.mean():>9.3f}{i0.mean():>9.3f}{b.mean():>9.3f}"
              f"{art.mean():>+10.3f}{net.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]")

    print("\n해석:")
    print("  **순효과 = 보정 후 − 항등**이 판정 대상이다. 항등 통제는 참 효과가 0 인 조건이므로,")
    print("  거기서 나오는 증가분(아티팩트 열)은 리샘플링이 만든 인공 운동이다.")
    print("  순효과의 CI 가 0 을 포함하거나 양수면 자기 운동 제거 효과가 없다는 뜻이고,")
    print("  뚜렷한 음수면 정적 스트림이 되살아난다는 뜻이다.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"_meta": {"n_per_dataset": args.n, "n_frames": N_FRAMES,
                             "tau": MAG_THRESHOLD, "seed": args.seed,
                             "method": "warp-then-flow: goodFeaturesToTrack + LK + estimateAffinePartial2D(RANSAC), then Farneback on aligned pair"},
                   "summary": summary, "clips": results}, f, ensure_ascii=False, indent=2)
    print(f"\n기록: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
