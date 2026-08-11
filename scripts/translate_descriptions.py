#!/usr/bin/env python
"""정답 설명을 한국어로 미리 번역해 둔다 (라벨링 UI의 확인 단계용).

주의 — 번역은 **판정에 쓰이지 않는다.** 라벨링 UI는 영상만 보고 판정하게 하고, 정답
설명(영문 + 한국어)은 판정을 마친 뒤 확인용으로만 보여준다. 먼저 보여주면 답을 알고
판정하게 되어 일치도(κ) 측정이 무의미해지기 때문이다.

학습·평가는 전부 영문 원문으로 이루어지며, 번역본은 어디에도 입력되지 않는다.

사용법:
    $CERBERUS_PY scripts/translate_descriptions.py --queue-only    # 라벨링 대상만 (권장)
    $CERBERUS_PY scripts/translate_descriptions.py --limit 100
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
ANNO = f"{ROOT}/hawk_anomaly/Annotation/All_Mix/all_videos_all.local.json"
QUEUE = f"{ROOT}/hawk/experiments/bg_critical_benchmark/annotation_queue.json"
OUT = f"{ROOT}/hawk/experiments/bg_critical_benchmark/descriptions_ko.json"

PROMPT = """다음은 CCTV·블랙박스 영상에 대한 영문 설명입니다. 한국어로 번역하세요.

규칙:
- 사실만 옮기고 해석이나 추측을 덧붙이지 마십시오.
- 장소·노면 상태·기상·조명 같은 **장면 묘사를 빠뜨리지 마십시오**.
- 자연스러운 한국어로 쓰되 원문에 없는 내용을 만들지 마십시오.
- 번역문만 출력하고 다른 말은 하지 마십시오.

영문: {text}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--queue-only", action="store_true",
                    help="라벨링 대기열에 있는 클립만 번역 (기본 권장 — 전수는 불필요)")
    # UI 가 실제로 보여줄 표본만 번역하면 API 호출을 아낀다. annotate_ui.py 와 같은
    # --sample/--seed 를 주면 정확히 같은 집합이 선택된다.
    ap.add_argument("--sample", type=int, default=0,
                    help="대기열에서 무작위 N건만 (annotate_ui.py 의 --sample 과 맞출 것)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32,
                    help="동시 요청 수. 429가 잦으면 낮출 것")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import os as _os
    from google import genai
    from google.genai import types

    key = _os.environ.get("GEMINI_API_KEY")
    if not key and os.path.exists(f"{ROOT}/.gemini_token"):
        key = open(f"{ROOT}/.gemini_token").read().strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY 가 없습니다.")

    with open(ANNO) as f:
        samples = {s["video"]: s.get("description", "") for s in json.load(f)}

    targets = list(samples)
    if args.queue_only and os.path.exists(QUEUE):
        with open(QUEUE) as f:
            queue = json.load(f)
        if args.sample:
            # annotate_ui.py 와 동일한 추출 (같은 seed, 같은 리스트 → 같은 집합)
            import random as _r
            queue = _r.Random(args.seed).sample(queue, min(args.sample, len(queue)))
        targets = [it["video_path"] for it in queue]
    if args.limit:
        targets = targets[: args.limit]

    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
        print(f"[재개] 기존 번역 {len(out)}건")

    client = genai.Client(api_key=key)
    todo = [t for t in targets if t not in out and samples.get(t)]
    print(f"번역 대상 {len(todo)}건 (모델 {args.model}, 동시 {args.workers})")

    # API 호출은 I/O 대기가 대부분이라 직렬로 돌리면 대기 시간만 쌓인다.
    # 스레드 풀로 동시에 보내되, 429(속도 제한)는 지수 백오프로 재시도한다.
    def translate_one(vid):
        delay = 2.0
        for attempt in range(5):
            try:
                r = client.models.generate_content(
                    model=args.model,
                    contents=PROMPT.format(text=samples[vid]),
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return vid, r.text.strip(), None
            except Exception as exc:
                msg = str(exc)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return vid, None, msg[:120]
        return vid, None, "재시도 소진"

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(translate_one, v): v for v in todo}
        for fut in as_completed(futures):
            vid, text, err = fut.result()
            if text:
                out[vid] = text
            else:
                failed += 1
                if failed <= 3:
                    print(f"  [실패] {vid}: {err}")
            done += 1
            if done % 25 == 0 or done == len(todo):
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"  {done}/{len(todo)} 저장 (실패 {failed})")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"완료 — {args.out} ({len(out)}건, 실패 {failed})")


if __name__ == "__main__":
    main()
