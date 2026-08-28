#!/usr/bin/env python
"""배경 분기가 자기 입력에 반응하는가를 직접 잰다.

왜 필요한가
-----------
`zero` arm 이 `flow` arm 과 같거나 낫다는 결과가 세 지표에서 반복됐다. 두 가지 설명이
가능하다.
  (A) 배경 분기가 입력을 무시하도록 학습됐다 — 픽셀이 어디로도 전달되지 않는다.
  (B) 정보는 전달되는데 LLM 이 쓰지 않는다.
(A) 라면 Stage-1 을 고쳐도 같은 붕괴가 재현될 공산이 크고, (B) 라면 Stage-1 의 정렬 결함이
원인일 수 있어 재학습에 걸어볼 만하다. 3주짜리 결정의 근거이므로 학습 없이 먼저 잰다.

무엇을 재나
-----------
같은 클립에 정적 입력만 flow/zero/random_mask/duplicate 로 바꿔 넣고 배경 분기의 출력
임베딩이 얼마나 달라지는지 본다. 비교 기준(scale)이 필요하므로 **외형 분기**를 같은 방식으로
잰다 — 외형 분기는 입력이 바뀌면 당연히 크게 변해야 하는 정상 동작의 기준선이다.

  across_input : 같은 클립, 입력 모드만 다름 → 낮을수록 입력을 무시
  across_clip  : 같은 입력 모드, 클립만 다름 → 낮으면 출력이 상수로 붕괴
"""
import argparse, os, sys, json, itertools
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _u
_s = _u.spec_from_file_location("_ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
_ev = _u.module_from_spec(_s)
try: _s.loader.exec_module(_ev)
except SystemExit: pass

from hawk.processors.video_processor import load_streams_aligned, apply_shared_transform

MODES = ["flow", "zero", "random_mask", "duplicate"]


def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu 로 주면 CPU 에서 돈다 (GPU 가 학습으로 만석일 때)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _ev.enforce_determinism(0)
    dev = a.device or f"cuda:{a.gpu_id}"
    chat = _ev.build_chat(a.cfg, a.ckpt, a.gpu_id, use_background=True, device=a.device)
    model = chat.model

    anno = json.load(open(a.anno))
    vids, seen = [], set()
    for r in (anno if isinstance(anno, list) else anno.get("annotations", [])):
        v = r.get("video") or r.get("video_path") or r.get("video_id")
        if not v or v in seen: continue
        p = v if os.path.isabs(v) else os.path.join(_ev.ROOT, "hawk_anomaly/Videos", v)
        if os.path.exists(p) and os.path.getsize(p) < 2 * 2**30:
            seen.add(v); vids.append(p)
        if len(vids) >= a.n: break
    print(f"  클립 {len(vids)}개")

    emb = {m: {"bg": [], "app": []} for m in MODES}
    for i, p in enumerate(vids, 1):
        for m in MODES:
            try:
                v, vm, vb = load_streams_aligned(p, n_frms=32, image_size=224,
                                                 sampling="uniform", ablation=m)
                v, vm, vb = apply_shared_transform(chat.vis_processor.transform, (v, vm, vb))
                v = v.unsqueeze(0).to(dev); vb = vb.unsqueeze(0).to(dev)
                with torch.no_grad():
                    eb, _, _ = model.encode_videoQformer_visual(vb, background=True)
                    ea, _, _ = model.encode_videoQformer_visual(v)
                emb[m]["bg"].append(eb.float().cpu().numpy())
                emb[m]["app"].append(ea.float().cpu().numpy())
            except Exception as e:
                print(f"  [실패] {os.path.basename(p)} {m}: {e}")
                for k in ("bg", "app"): emb[m][k].append(None)
        if i % 10 == 0: print(f"  {i}/{len(vids)}", flush=True)

    ok = [i for i in range(len(vids))
          if all(emb[m]["bg"][i] is not None for m in MODES)]
    print(f"  유효 클립 {len(ok)}개")

    res = {"n": len(ok), "ckpt": a.ckpt, "across_input": {}, "across_clip": {}}
    for br in ("bg", "app"):
        # 같은 클립, 입력 모드만 다름
        for m1, m2 in itertools.combinations(MODES, 2):
            s = [cos(emb[m1][br][i], emb[m2][br][i]) for i in ok]
            res["across_input"][f"{br}:{m1}-{m2}"] = round(float(np.mean(s)), 4)
        # 같은 입력, 클립만 다름
        for m in MODES:
            pairs = [cos(emb[m][br][i], emb[m][br][j])
                     for i, j in itertools.combinations(ok[:40], 2)]
            res["across_clip"][f"{br}:{m}"] = round(float(np.mean(pairs)), 4)

    json.dump(res, open(a.out, "w"), indent=2)
    print("\n=== 같은 클립, 입력 모드만 변경 (1.0 = 완전히 무시) ===")
    for k, v in res["across_input"].items(): print(f"  {k:<28} {v:.4f}")
    print("\n=== 같은 입력, 클립만 변경 (1.0 = 출력이 상수로 붕괴) ===")
    for k, v in res["across_clip"].items(): print(f"  {k:<28} {v:.4f}")


if __name__ == "__main__":
    main()
