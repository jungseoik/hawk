#!/usr/bin/env python
"""Stage-2 배경 감독(`L_BL`)용 문장을 미리 만들어 둔다.

왜 필요한가
-----------
Stage-2 의 forward 는 `loss_background` 로 **메인 손실을 그대로 되돌려준다**
(`video_llama.py`, `conv_type=='multi'` 경로). 즉 배경 분기에 전용 감독이 없다.
그 결과 분기가 상수로 붕괴한다 — 클립이 달라도 출력 코사인 0.998,
정상인 외형 분기는 0.76 (실측, `diag_branch_sensitivity.py`).
붕괴는 랜덤 초기화에서 다시 학습해도 재현되므로(`flow_reinit` 0.9994) Stage-1 탓이 아니다.

배경 토큰만으로 장면 문장을 생성하게 만들면 분기가 자기 입력을 볼 이유가 생긴다.
그 문장을 여기서 만든다 — Stage-1 이 쓰는 추출기와 **같은 것**을 써서 두 단계의
감독 대상을 일치시킨다.

학습 중에 spacy 를 돌리면 느리므로 오프라인으로 한 번만 만들어 둔다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hawk.datasets.datasets.webvid_datasets import extract_background_entities_sentence

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = json.load(open(a.anno))
    out, empty = {}, 0
    for i, r in enumerate(recs, 1):
        desc = (r.get("description") or "").strip()
        key = os.path.basename(r.get("video", ""))
        if not desc or not key:
            continue
        s = extract_background_entities_sentence(desc)
        # 추출 실패 시 원문을 되돌려주는 구현이라, 그대로면 감독이 배경에 한정되지 않는다.
        if s.strip() == desc:
            empty += 1
        out[key] = s
        if i % 1000 == 0:
            print(f"  {i}/{len(recs)}", flush=True)

    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    n = len(out)
    lens = [len(v.split(", ")) for v in out.values()]
    uniq = len(set(out.values()))
    print(f"  저장 {n}건 → {a.out}")
    print(f"  추출 실패(원문 그대로) {empty}건 · 항목 수 중앙 {sorted(lens)[n//2]} · 서로 다른 문장 {uniq}건 ({100*uniq/n:.1f}%)")
    for k in list(out)[:3]:
        print(f"    {k[:34]:<34} → {out[k][:90]}")
