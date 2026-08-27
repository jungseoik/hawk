#!/usr/bin/env python
"""HAWK 자체 지표(Reasonability/Detail/Consistency)를 기존 평가 결과에 매긴다.

왜 별도 스크립트인가
--------------------
`evaluate.py --judge` 는 생성부터 다시 돌린다(모델당 2.5시간, GPU 필요). 판정은 이미 나온
문장을 채점하는 일이므로 결과 JSON 만 있으면 된다 — GPU 0, API 호출뿐.

⚠ 판정자 절대값을 원본 HAWK 논문 표와 나란히 놓지 말 것. 그쪽은 GPT 가 매긴 점수이고
여기는 Gemini 다. **대신 HAWK 공개 체크포인트를 우리가 같은 판정자로 채점**하므로,
`hawk_official` 행과의 비교는 동일 조건이고 그것이 이 스크립트의 목적이다.

클립별 점수를 남기므로 `compare_arms.py` 와 같은 쌍 부트스트랩을 적용할 수 있다.
"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _u
_spec = _u.spec_from_file_location("_ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
_ev = _u.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_ev)
except SystemExit:
    pass

DIMS = ("reasonability", "detail", "consistency")


def make_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        p = "/home/work/seoik/.gemini_token"
        if os.path.exists(p):
            key = open(p).read().strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY 없음")
    return genai.Client(api_key=key)


def judge_records(records, model, workers=8, limit=0, retries=3):
    from google.genai import types
    client = make_client()
    subset = records[:limit] if limit else records
    lock, state = threading.Lock(), {"done": 0, "fail": 0}

    def one(rec):
        gt, pred = rec.get("gt_description") or "", rec.get("pred_description") or ""
        if not gt or not pred:
            return
        for _ in range(retries):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=_ev.JUDGE_PROMPT.format(reference=gt, hypothesis=pred),
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                m = re.search(r"\{.*\}", r.text.strip(), re.S)
                s = json.loads(m.group(0) if m else r.text.strip())
                rec["judge_scores"] = {d: float(s[d]) for d in DIMS}
                break
            except Exception:
                continue
        with lock:
            state["done"] += 1
            if "judge_scores" not in rec:
                state["fail"] += 1
            if state["done"] % 100 == 0:
                print(f"    {state['done']}/{len(subset)} (실패 {state['fail']})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, subset))

    scored = [r for r in subset if "judge_scores" in r]
    if not scored:
        raise RuntimeError("판정 전부 실패")
    return ({d: sum(r["judge_scores"][d] for r in scored) / len(scored) for d in DIMS},
            len(scored), state["fail"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    for p in a.eval:
        d = json.load(open(p))
        if d.get("metrics", {}).get("judge") and not a.limit:
            print(f"  {os.path.basename(p)}: 이미 판정됨 — 건너뜀"); continue
        print(f"  {os.path.basename(p)} 판정 중 …", flush=True)
        means, n, fail = judge_records(d["records"], a.model, a.workers, a.limit)
        line = "  ".join(f"{k}={v:.4f}" for k, v in means.items())
        print(f"    {line}  (n={n}, 실패 {fail})")
        if not a.limit:
            d.setdefault("metrics", {})["judge"] = {**means, "_model": a.model,
                                                    "_n": n, "_failed": fail}
            json.dump(d, open(p, "w"), ensure_ascii=False)
