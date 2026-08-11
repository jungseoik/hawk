#!/usr/bin/env python
"""Background-critical 라벨링 — LLM 판정자 (Gemini).

사람 주석자와 **완전히 같은 절차**를 따른다. 그래야 두 라벨의 일치도(κ)가
"판정자 간 차이"를 재는 값이 된다.

  1단계  동적 프레임만 (M ⊙ x, 배경이 검게 지워짐) → "이상을 특정할 수 있는가?"
  2단계  원본 프레임                                → 같은 질문
  라벨   두 답변에서 기계적으로 도출 (annotation_protocol.md §2.1)

설계상 지킨 것
--------------
- **가설을 알리지 않는다.** 프롬프트에 배경 스트림·상보 분해·연구 주장이 등장하지 않는다.
  판정자가 "배경이 중요하다고 답해야 한다"는 압력을 받으면 라벨이 오염된다.
- **모델 출력을 보여주지 않는다.** CERBERUS 의 생성 결과는 프롬프트에 들어가지 않는다.
- **정답 캡션을 주지 않는다.** 캡션의 장면 어휘로 판정하면, 우리 모델이 생성하도록 학습된
  바로 그 어휘가 라벨을 결정하게 되어 순환이 된다.
- 온도 0, 모델 버전·프레임 수를 결과에 기록해 재현 가능하게 한다.

사용법:
    $CERBERUS_PY scripts/annotate_bg_critical_llm.py --sample 150 --seed 42   # κ 측정용
    $CERBERUS_PY scripts/annotate_bg_critical_llm.py --sample 0 --workers 12  # 전수
"""
import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
QUEUE = f"{ROOT}/hawk/experiments/bg_critical_benchmark/annotation_queue.json"
VIDEOS = f"{ROOT}/hawk_anomaly/Videos"
OUT = f"{ROOT}/hawk/experiments/bg_critical_benchmark/labels_llm.json"
MAG_THRESHOLD = 0.2

# v1 (유도 질문 — 보존용). "이상을 특정할 수 있는가"로 물어 이상의 존재를 전제했고,
# 그 결과 정상 클립의 46%에서 무언가를 찾아냈다(오탐). 비교 기록으로 남긴다.
QUESTION_V1 = """You are shown {n} frames sampled in order from a surveillance or dashcam video.

{context}

Question: from these frames alone, can you identify a specific anomalous event —
something unsafe, unusual, or out of place — and say what it is?

Answer with JSON only:
{{"answer": <1|2|3|4>, "what": "<one short phrase, or empty>"}}

  1 = yes, I can identify a specific anomalous event
  2 = something seems off, but I cannot pin down what
  3 = no, nothing anomalous — this looks like ordinary activity
  4 = I cannot judge; the frames do not show enough to tell

Choose 3 only when the scene genuinely looks ordinary. Choose 4 only when the
frames are too fragmentary or unclear to make any judgement."""

# v2 — 중립 프레이밍. 세 가지를 고쳤다.
#  (1) 이상의 존재를 전제하지 않고, 먼저 "무슨 일이 일어나는가"를 서술하게 한다.
#  (2) 대다수 클립이 평범하다는 사실을 알려 3번이 기본값이 되게 한다.
#  (3) 이상이 아닌 것을 명시적으로 열거한다 — v1 이 오탐한 실제 사례에서 뽑았다
#      (합성 영상의 렌더링 어색함, 가만히 서 있는 사람, 평범한 차량 등장).
QUESTION_V2 = """You are shown {n} frames sampled in order from a video.

{context}

Most clips in this collection show ordinary, uneventful activity. Some contain a
genuine incident. Your job is to report what you actually see — not to find
something.

First describe briefly what is happening. Then decide.

Do NOT count as an incident:
- ordinary movement (people walking, standing, sitting; vehicles driving or parked)
- an unusual camera angle, low resolution, compression artifacts, odd lighting
- anything that merely looks synthetic, rendered, or visually strange
- a scene you find unfamiliar but where nothing harmful happens

Count as an incident only a concrete event a person would report: a collision,
a fall, a fire, a fight, a theft, someone or something where it is clearly unsafe
to be.

Answer with JSON only:
{{"happening": "<one short phrase>", "answer": <1|2|3|4>}}

  1 = a specific incident occurs, and I can name it
  2 = something may be wrong but I cannot tell what — genuinely uncertain
  3 = nothing of the sort; this is ordinary activity
  4 = the frames are too fragmentary to judge at all

3 is the expected answer for most clips. Use 2 when you are truly unsure rather
than forcing a choice between 1 and 3."""

QUESTIONS = {"v1": QUESTION_V1, "v2": QUESTION_V2}

CTX_MOTION = ("In these frames, everything that was not moving has been blacked out. "
              "Only the moving parts of each frame remain visible.")
CTX_FULL = "These are the unmodified frames."


def extract_frames(video_path, n_frames):
    """원본 프레임과 동적 프레임을 학습 파이프라인과 같은 방식으로 만든다."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return [], []
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    originals, motions = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(i) - 1, 0))
        ok_p, prev = cap.read()
        ok_c, cur = cap.read()
        if not (ok_p and ok_c):
            continue
        prev_s, cur_s = cv2.resize(prev, (448, 448)), cv2.resize(cur, (448, 448))
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_s, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(cur_s, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = (mag > MAG_THRESHOLD).astype(np.uint8)[..., None]
        originals.append(cur_s)
        motions.append(cur_s * mask)
    cap.release()
    return originals, motions


def _jpegs(frames):
    out = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(buf.tobytes())
    return out


def derive_label(a1, a2):
    """annotation_protocol.md §2.1 과 동일한 도출 규칙 (사람 UI 와 같은 코드 경로 의미)."""
    if a1 == 4 or a2 == 4:
        return "unjudgeable"
    if a2 == 3:
        return "normal"
    if a1 == 1:
        return "motion_sufficient"
    if a2 == 1:
        return "context_critical"
    return "context_dependent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--queue", default=QUEUE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sample", type=int, default=150,
                    help="annotate_ui.py 와 같은 값을 주면 같은 집합 (κ 측정용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", choices=("v1", "v2"), default="v2",
                    help="v1=초기(유도 질문, 정상 오탐 46%%) / v2=중립 프레이밍")
    ap.add_argument("--frames", type=int, default=16,
                    help="판정에 쓸 프레임 수. 동적 스트림은 시간 신호라 장수가 적으면 읽기 어렵다")
    ap.add_argument("--workers", type=int, default=32,
                    help="동시 요청 수. 실측: 클립당 CPU 1.6초 / API 대기 11초로 I/O 바운드라 "
                         "올릴수록 선형에 가깝게 빨라진다. 429가 잦으면 낮출 것")
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key and os.path.exists(f"{ROOT}/.gemini_token"):
        key = open(f"{ROOT}/.gemini_token").read().strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY 가 없습니다.")
    client = genai.Client(api_key=key)

    with open(args.queue) as f:
        items = json.load(f)
    if args.sample:
        items = random.Random(args.seed).sample(items, min(args.sample, len(items)))

    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
        print(f"[재개] 기존 라벨 {len(out)}건")
    todo = [it for it in items if it["clip_id"] not in out]
    print(f"라벨 대상 {len(todo)}건 (모델 {args.model}, 프레임 {args.frames}, 동시 {args.workers})")

    def ask(jpegs, context):
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpegs]
        parts.append(types.Part.from_text(
            text=QUESTIONS[args.prompt].format(n=len(jpegs), context=context)))
        delay = 2.0
        for attempt in range(5):
            try:
                r = client.models.generate_content(
                    model=args.model, contents=parts,
                    config=types.GenerateContentConfig(temperature=0.0))
                import re
                m = re.search(r"\{.*\}", r.text, re.S)
                d = json.loads(m.group(0) if m else r.text)
                what = d.get("what") or d.get("happening") or ""
                return int(d["answer"]), str(what)[:120]
            except Exception as exc:
                msg = str(exc)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 4:
                    time.sleep(delay); delay *= 2; continue
                raise
        raise RuntimeError("재시도 소진")

    def label_one(item):
        path = os.path.join(VIDEOS, item["video_path"])
        try:
            originals, motions = extract_frames(path, args.frames)
            if not originals:
                return item["clip_id"], None, "프레임 추출 실패"
            a1, w1 = ask(_jpegs(motions), CTX_MOTION)     # 1단계: 배경 제거
            a2, w2 = ask(_jpegs(originals), CTX_FULL)     # 2단계: 원본
            return item["clip_id"], {
                "motion_sufficiency": derive_label(a1, a2),
                "a_motion_only": a1, "a_original": a2,
                "what_motion_only": w1, "what_original": w2,
                "annotator": args.model, "n_frames": args.frames,
                "prompt_version": args.prompt,
            }, None
        except Exception as exc:
            return item["clip_id"], None, str(exc)[:120]

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(label_one, it) for it in todo]
        for fut in as_completed(futures):
            cid, rec, err = fut.result()
            if rec:
                out[cid] = rec
            else:
                failed += 1
                if failed <= 3:
                    print(f"  [실패] {cid}: {err}")
            done += 1
            if done % 20 == 0 or done == len(todo):
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"  {done}/{len(todo)} 저장 (실패 {failed})")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"완료 — {args.out} ({len(out)}건, 실패 {failed})")


if __name__ == "__main__":
    main()
