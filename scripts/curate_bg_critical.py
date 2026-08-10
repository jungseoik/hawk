#!/usr/bin/env python
"""Background-critical 벤치마크의 **주석 후보**를 선별한다.

7,852개 클립을 전부 사람이 주석하는 것은 비현실적이므로, 모델과 무관한 두 신호로
순위를 매겨 주석 순서를 정한다. 라벨 자체는 사람이 붙인다 — 이 스크립트는 라벨을
만들지 않으며, 만들어서도 안 된다 (`experiments/bg_critical_benchmark/annotation_protocol.md` §4).

두 신호 (둘 다 정답 캡션과 광학 플로우에서만 나오고, 어떤 모델 출력도 쓰지 않는다):

  1. 정적 우세도 = 1 − motion mask 비율
     플로우 마스크가 거의 비어 있는데 이상이 보고된 클립은 `context_critical` 후보다.
  2. 장면 어휘 밀도 = 정답 설명에서 장면 어휘(비-논항 명사 + 형용사)가 차지하는 비율
     설명이 노면·기상·장소를 많이 언급할수록 맥락 의존 후보다.

선별 편향을 점검할 수 있도록 **무작위 대조 표본**도 함께 뽑는다. 상위 후보와 무작위
표본의 `context_critical` 비율을 비교하면 선별이 놓친 양을 추정할 수 있다.

사용법:
    $CERBERUS_PY scripts/curate_bg_critical.py --workers 24 --limit 0
    $CERBERUS_PY scripts/curate_bg_critical.py --limit 200      # 빠른 시험
"""
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
ANNO = f"{ROOT}/hawk_anomaly/Annotation/All_Mix/all_videos_all.local.json"
VIDEOS = f"{ROOT}/hawk_anomaly/Videos"
OUT_DIR = f"{ROOT}/hawk/experiments/bg_critical_benchmark"

MAG_THRESHOLD = 0.2   # video_processor.py 와 동일해야 한다
N_PROBE_FRAMES = 8


def motion_ratio(video_path, n_frms=N_PROBE_FRAMES):
    """클립의 평균 motion-mask 비율. 학습 파이프라인과 같은 설정을 쓴다."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frms + 1).astype(int)
    ratios, prev_gray = [], None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (224, 224))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 10, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ratios.append(float((mag > MAG_THRESHOLD).mean()))
        prev_gray = gray

    cap.release()
    return float(np.mean(ratios)) if ratios else None


def scene_word_density(description):
    """정답 설명에서 장면 어휘가 차지하는 비율.

    학습에 쓰이는 것과 동일한 추출기를 재사용해, 선별 기준과 감독 신호가 어긋나지
    않게 한다.
    """
    from hawk.datasets.datasets.webvid_datasets import (
        extract_background_entities_sentence,
    )

    words = description.split()
    if not words:
        return 0.0
    scene = extract_background_entities_sentence(description)
    # 추출 실패 시 원문을 그대로 돌려주는 fallback 이 있으므로, 그 경우는 0으로 본다.
    if scene.strip() == description.strip():
        return 0.0
    return len(scene.split()) / len(words)


def _probe(record):
    path = os.path.join(VIDEOS, record["video"])
    try:
        ratio = motion_ratio(path)
    except Exception as exc:                      # 개별 클립 실패가 전체를 막지 않도록
        return {"video": record["video"], "error": str(exc)}
    if ratio is None:
        return {"video": record["video"], "error": "unreadable"}
    return {
        "video": record["video"],
        "source_dataset": record["video"].split("/")[0],
        "motion_ratio": ratio,
        "static_dominance": 1.0 - ratio,
        "scene_density": scene_word_density(record.get("description", "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default=ANNO)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--top", type=int, default=400, help="주석 후보 상위 N")
    ap.add_argument("--control", type=int, default=200, help="무작위 대조 표본 크기")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.anno) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    print(f"클립 {len(records)}개 분석 (workers={args.workers})")

    results, failed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_probe, r): r for r in records}
        for i, fut in enumerate(as_completed(futures), 1):
            out = fut.result()
            if "error" in out:
                failed += 1
            else:
                results.append(out)
            if i % 500 == 0:
                print(f"  {i}/{len(records)}  (실패 {failed})")

    print(f"완료: 성공 {len(results)}, 실패 {failed}")

    # 데이터셋별 마스크 비율 — 논문 §4 에 실릴 표이자 H1 검증의 전제
    print("\n=== 데이터셋별 motion-mask 비율 ===")
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["source_dataset"], []).append(r["motion_ratio"])
    for ds, vals in sorted(by_ds.items()):
        arr = np.array(vals)
        print(f"  {ds:<14} n={len(arr):<5} mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
              f"p90={np.percentile(arr, 90):.3f}")

    # 후보 순위: 정적 우세도와 장면 어휘 밀도를 각각 순위화한 뒤 합산
    def ranks(key):
        order = sorted(range(len(results)), key=lambda i: results[i][key])
        rank = [0] * len(results)
        for pos, idx in enumerate(order):
            rank[idx] = pos / max(len(results) - 1, 1)
        return rank

    r_static, r_scene = ranks("static_dominance"), ranks("scene_density")
    for i, r in enumerate(results):
        r["candidate_score"] = round((r_static[i] + r_scene[i]) / 2, 4)

    ranked = sorted(results, key=lambda r: -r["candidate_score"])
    top = ranked[: args.top]

    rng = random.Random(args.seed)
    pool_ids = {r["video"] for r in top}
    control = rng.sample([r for r in results if r["video"] not in pool_ids],
                         min(args.control, max(len(results) - len(top), 0)))

    os.makedirs(OUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUT_DIR, "mask_statistics.json")
    with open(stats_path, "w") as f:
        json.dump({ds: {"n": len(v), "mean": float(np.mean(v)),
                        "median": float(np.median(v))} for ds, v in by_ds.items()}, f, indent=2)

    # 주석용 매니페스트 뼈대 — 라벨 필드는 비워 둔다. 사람이 채운다.
    def skeleton(r, group):
        return {
            "clip_id": r["video"].replace("/", "__"),
            "source_dataset": r["source_dataset"],
            "video_path": r["video"],
            "selection_group": group,          # candidate / random_control
            "candidate_score": r["candidate_score"],
            "motion_ratio": round(r["motion_ratio"], 4),
            "scene_density": round(r["scene_density"], 4),
            "motion_sufficiency": None,        # ← 주석자가 채운다
            "ego_motion": None,                # ← 주석자가 채운다
            "annotator": None,
        }

    manifest = [skeleton(r, "candidate") for r in top] + \
               [skeleton(r, "random_control") for r in control]
    manifest_path = os.path.join(OUT_DIR, "annotation_queue.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n주석 대기열: {manifest_path}")
    print(f"  후보 {len(top)} + 무작위 대조 {len(control)} = {len(manifest)}")
    print(f"마스크 통계: {stats_path}")
    print("\n다음 단계: annotation_protocol.md 에 따라 2인 이상이 "
          "motion_sufficiency / ego_motion 을 독립적으로 채우고 κ 를 계산할 것.")


if __name__ == "__main__":
    main()
