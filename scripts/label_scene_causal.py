#!/usr/bin/env python
"""정답 설명에서 **정적 장면 조건이 원인 설명의 일부인가**를 판정한다.

왜 이 방식으로 바꿨나
---------------------
기존 프로토콜은 사람에게 (1) 배경이 지워진 영상, (2) 원본 영상을 차례로 보여주고
"모션만으로 판별 가능한가"를 묻는 2단계 절차였다. 실측 결과 사람–LLM 일치도가
Cohen's κ = 0.383에 머물렀고, 원인이 분명했다 — **사람과 LLM이 서로 다른 정보를 보고
판정했다.** 사람은 전경이 지워진 저정보 영상을, LLM은 프레임을 봤다. 같은 것을 보지
않는 두 판정자의 일치도는 절차의 신뢰도가 아니라 절차의 결함을 재고 있었다.

이 스크립트는 판정 대상을 **정답 설명 텍스트**로 옮긴다. 세 가지 이유에서 이게 낫다.

1. **평가와 일치한다.** 우리는 생성문을 정답 설명과 비교해 채점한다. 정답이 "폭우로
   시야를 잃어" 사고가 났다고 쓰면, 비를 놓친 모델이 실제로 점수를 잃는다. 반면 기존
   정의("사람이 모션만 보고 판별 가능한가")는 채점에 쓰이지 않는 반사실적 판단이었다.
2. **두 판정자가 같은 것을 본다.** 사람과 LLM이 동일한 텍스트를 읽으므로, 일치도가
   판정 기준의 명확성을 반영한다.
3. **비용이 낮다.** 사람은 471건 전량이 아니라 신뢰도 검증용 표본만 읽으면 된다.

판정 범주
---------
`causal`      정적 장면 조건(노면·기상·조명·장소)이 **원인·위험도 설명의 일부**다.
              예: "폭우로 와이퍼가 시야를 확보하지 못해 추돌했다"
`incidental`  장면이 언급되나 **배경으로만** 등장한다. 원인은 행위·객체에 있다.
              예: "도심 도로를 달리다 앞차가 감속한 것을 못 보고 추돌했다"
              ("도심 도로"는 장소일 뿐 원인이 아니다)
`no_scene`    정적 장면 조건이 설명에 등장하지 않는다.
`normal`      설명에 이상 사건이 없다(정상 클립).

`causal`이 벤치마크의 핵심 부분집합이고, `incidental`과의 구분이 이 판정의 요점이다.
자동 어휘 매칭으로는 이 둘이 갈리지 않는다 — heldout 471건에서 장면 어휘가 등장하는
클립은 424건(90%)이지만 그중 상당수가 `incidental`이다.

사용
----
    # LLM 판정 (전량)
    $CERBERUS_PY scripts/label_scene_causal.py \
        --anno /home/work/seoik/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json \
        --out experiments/bg_critical_benchmark/labels_scene_causal_llm.json

    # 사람 검증용 표본 추출 (무작위, 시드 고정)
    $CERBERUS_PY scripts/label_scene_causal.py --anno ... --sample 80 --seed 20260811 \
        --out experiments/bg_critical_benchmark/validation_sample.json --dry-run
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
MODEL = "gemini-3.1-pro-preview"

# 판정 프롬프트. 유도적 표현을 피하고 `incidental`을 **먼저** 제시해,
# "장면이 나오면 원인이다"로 기울지 않게 한다. 초기 프롬프트가 이상을 찾도록
# 유도해 정상 클립 오탐이 46%까지 올랐던 경험을 반영한 설계다.
PROMPT = """You are analysing a reference description of a video, written by a human annotator.

Your task: decide whether a STATIC SCENE CONDITION is part of the *causal or risk explanation*, or whether it is merely the setting.

Static scene conditions are properties of the place that do not move: road surface state (wet, icy, gravel), weather (rain, fog, snow), lighting (dark, dim, glare), place type where that type itself creates the hazard (highway lane, tunnel, construction zone), and crowding.

Choose exactly one:

- "incidental": a scene or place is mentioned, but the cause lies in what someone or something DID. The scene could be swapped for another without changing the explanation.
  Example: "While driving through the city street, the vehicle fails to notice the traffic slowing ahead and rear-ends the car in front."
  (The street is where it happened, not why.)

- "causal": a static scene condition is presented as contributing to why the event happened or why it was dangerous. Removing that condition would change the explanation.
  Example: "During the heavy rain, the wipers cannot keep up and the driver loses visibility, causing a rear-end collision."
  (The rain is why.)

- "no_scene": no static scene condition appears in the description at all.

- "normal": the description does not describe any anomalous, dangerous, or unusual event.

Judge only what the text states. Do not infer conditions that are not written.

Reference description:
---
{description}
---

Respond with JSON only, no prose:
{{"label": "<incidental|causal|no_scene|normal>", "scene_terms": ["<terms you based this on, or empty>"], "reason": "<one short sentence>"}}"""

LABELS = ("incidental", "causal", "no_scene", "normal")


def _key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        p = f"{ROOT}/.gemini_token"
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        raise RuntimeError(f"GEMINI_API_KEY 또는 {ROOT}/.gemini_token 이 필요합니다.")
    return k


def judge_one(client, rec, model, max_retry=5):
    from google.genai import types

    desc = (rec.get("description") or "").strip()
    if not desc:
        return {"label": None, "error": "설명 없음"}

    for attempt in range(max_retry):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=PROMPT.format(description=desc),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            m = re.search(r"\{.*\}", resp.text, re.S)
            if not m:
                return {"label": None, "error": f"JSON 없음: {resp.text[:80]}"}
            out = json.loads(m.group(0))
            if out.get("label") not in LABELS:
                return {"label": None, "error": f"알 수 없는 라벨: {out.get('label')}"}
            return out
        except Exception as exc:
            msg = str(exc)
            # 429/503 은 지수 백오프로 재시도한다. 동시 요청을 쓰므로 이 처리가 없으면
            # 일시적 rate limit 이 영구 실패로 기록된다.
            if any(c in msg for c in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                if attempt < max_retry - 1:
                    time.sleep(2 ** attempt)
                    continue
            return {"label": None, "error": msg[:150]}
    return {"label": None, "error": "재시도 초과"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sample", type=int, default=0,
                    help="0이면 전량. N이면 무작위 N건만 (사람 검증 표본 추출용)")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--dry-run", action="store_true",
                    help="API를 호출하지 않고 대상 목록만 기록한다 (사람 검증 표본 준비)")
    args = ap.parse_args()

    with open(args.anno) as f:
        records = json.load(f)

    if args.sample:
        # 사람 검증 표본은 시드를 고정해 뽑는다. LLM 판정과 **같은 부분집합**에서
        # 일치도를 계산해야 하므로, 이 시드는 기록에 남고 바뀌지 않아야 한다.
        rng = random.Random(args.seed)
        records = rng.sample(records, min(args.sample, len(records)))
        print(f"무작위 표본 {len(records)}건 (seed={args.seed})")

    if args.dry_run:
        out = [{"clip_id": r["video"].replace("/", "__"),
                "video_path": r["video"],
                "source_dataset": r["video"].split("/")[0],
                "description": r["description"],
                "label": None,          # ← 사람이 채운다
                "annotator": None,
                "seconds_spent": None}
               for r in records]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"검증 표본 기록: {args.out} ({len(out)}건, 라벨 비어 있음)")
        return

    from google import genai
    client = genai.Client(api_key=_key())

    print(f"{len(records)}건 판정 (model={args.model}, workers={args.workers})")
    t0 = time.time()
    results = [None] * len(records)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_one, client, r, args.model): i
                for i, r in enumerate(records)}
        for fut in as_completed(futs):
            i = futs[fut]
            r = records[i]
            out = fut.result()
            results[i] = {
                "clip_id": r["video"].replace("/", "__"),
                "video_path": r["video"],
                "source_dataset": r["video"].split("/")[0],
                "description": r["description"],
                "annotator": args.model,
                **out,
            }
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(records)}  ({time.time() - t0:.0f}s)")

    ok = [r for r in results if r.get("label")]
    fail = len(results) - len(ok)
    dist = {k: sum(1 for r in ok if r["label"] == k) for k in LABELS}

    print(f"\n완료 {len(ok)}건, 실패 {fail}건  ({time.time() - t0:.0f}s)")
    print("=== 라벨 분포 ===")
    for k in LABELS:
        n = dist[k]
        print(f"  {k:<12} {n:>4}건 ({n / max(len(ok), 1):.1%})")
    print(f"\n벤치마크 핵심 부분집합(causal) = {dist['causal']}건")

    by_ds = {}
    for r in ok:
        by_ds.setdefault(r["source_dataset"], []).append(r["label"])
    print("\n=== 데이터셋별 causal 비율 (H1 예측의 공변량) ===")
    for ds, labs in sorted(by_ds.items()):
        c = sum(1 for x in labs if x == "causal")
        print(f"  {ds:<14} n={len(labs):<4} causal={c:<4} ({c / len(labs):.1%})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n기록: {args.out}")
    print("다음: --sample 로 뽑은 표본을 사람이 라벨하고 scripts/agreement.py 로 κ 계산")


if __name__ == "__main__":
    sys.exit(main())
