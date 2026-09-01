#!/usr/bin/env python
"""Charades 로 상보 분해(CVD)를 검증한다 — 배경 분기의 장면 정보량.

왜 Charades 인가
----------------
CVD 주장은 "정적 스트림에 `(1−M)⊙x` 를 넣어야 한다"이다. 이를 가르려면
  (a) 실제 움직임이 있어 마스크가 의미 있게 작동하고
  (b) 장면 라벨이 객관적이며
  (c) 우리 학습에 쓰인 적 없는
데이터가 필요하다. Charades test 는 셋 다 만족한다(1,863편, 장면 16종, 가정 내 활동 영상).

Places365 프로브는 정지 이미지라 마스크가 비어(M=0) `flow`≡`duplicate` 였다.
여기서는 마스크가 실제로 사람을 지우므로 둘이 갈린다.

측정
----
같은 모델에 정적 입력만 바꿔 넣고, 배경 분기 출력의 **평균을 뺀 잔차**로 장면을 검색한다.
(원 코사인은 큰 공통 성분 때문에 정보량과 무관하게 1 에 가까워진다 — 2026-08-31 정정 참조.)
"""
import argparse, json, os, sys, itertools, collections
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _u
_s = _u.spec_from_file_location("_ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
_ev = _u.module_from_spec(_s)
try: _s.loader.exec_module(_ev)
except SystemExit: pass
from hawk.processors.video_processor import load_streams_aligned, apply_shared_transform


def probe(X, Y):
    C = X - X.mean(0, keepdims=True)
    N = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = N @ N.T; np.fill_diagonal(S, -9)
    Y = np.array(Y); hit = (Y[S.argmax(1)] == Y).astype(int)
    cnt = collections.Counter(Y); n = len(Y)
    ch = float(np.mean([(cnt[y] - 1) / (n - 1) for y in Y]))
    raw = float(np.mean([X[i] @ X[j] / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-12)
                         for i, j in itertools.combinations(range(min(len(X), 80)), 2)]))
    return {"acc": float(hit.mean()), "chance": ch, "ratio": float(hit.mean()) / ch,
            "raw": raw, "n": n, "per_item": hit.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--ablation", default="flow")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _ev.enforce_determinism(0)
    chat = _ev.build_chat(a.cfg, a.ckpt, a.gpu_id, use_background=True)
    m, dev = chat.model, f"cuda:{a.gpu_id}"
    meta = json.load(open(a.subset))
    if a.limit: meta = meta[:a.limit]

    BG, AP, Y, ids = [], [], [], []
    for i, md in enumerate(meta, 1):
        try:
            v, vm, vb = load_streams_aligned(md["file"], n_frms=32, image_size=224,
                                             sampling="uniform", ablation=a.ablation)
            v, vm, vb = apply_shared_transform(chat.vis_processor.transform, (v, vm, vb))
            with torch.no_grad():
                eb, _, _ = m.encode_videoQformer_visual(vb.unsqueeze(0).to(dev), background=True)
                ea, _, _ = m.encode_videoQformer_visual(v.unsqueeze(0).to(dev))
            BG.append(eb.float().cpu().numpy().ravel())
            AP.append(ea.float().cpu().numpy().ravel())
            Y.append(md["name"]); ids.append(md["id"])
        except Exception as e:
            if i <= 3: print(f"  [실패] {md['id']}: {e}", flush=True)
        if i % 100 == 0: print(f"  {i}/{len(meta)}", flush=True)

    res = {"ablation": a.ablation, "ckpt": a.ckpt, "ids": ids,
           "배경": probe(np.array(BG), Y), "외형": probe(np.array(AP), Y)}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    for k in ("배경", "외형"):
        d = res[k]
        print(f"  [{a.ablation}] {k}: 잔차 장면검색 {d['acc']:.3f} "
              f"(우연 {d['chance']:.3f}, {d['ratio']:.1f}배, raw {d['raw']:.4f}, n={d['n']})")
