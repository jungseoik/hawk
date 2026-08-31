#!/usr/bin/env python
"""분기 정보량 진단 (중심화판) — 원 코사인 기반 진단의 오류를 바로잡는다.

⚠ 이전 진단(`diag_branch_sensitivity.py`)은 원 임베딩의 클립 간 코사인을 보고
"배경 분기가 상수로 붕괴했다"고 결론지었다. **그 결론은 틀렸다.**
임베딩에 큰 공통 성분이 있으면 원 코사인은 정보량과 무관하게 1 에 가까워진다.
평균을 빼고 보면 배경 분기 잔차는 장면을 우연 대비 6 배로 변별한다(Places365 실측).

여기서는 세 가지를 함께 보고한다.
  raw      : 원 임베딩 클립 간 코사인 (이전 지표 — 참고용으로만)
  centered : 표본 평균을 뺀 뒤의 클립 간 코사인 (0 근처면 정상)
  probe    : 잔차 최근접 이웃이 같은 장면인가 (정보량의 실질 척도)
"""
import argparse, json, os, re, sys, itertools, collections
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _u
_s = _u.spec_from_file_location("_ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
_ev = _u.module_from_spec(_s)
try: _s.loader.exec_module(_ev)
except SystemExit: pass
from hawk.processors.video_processor import load_streams_aligned, apply_shared_transform


def scene_of(p):
    m = re.search(r"ShanghaiTech/.*?/(\d\d)_\d+", p)
    if m: return "ST:" + m.group(1)
    m = re.search(r"UBnormal/(Scene\d+)/", p)
    if m: return "UB:" + m.group(1)
    return None


def stats(name, X, Y=None):
    raw = [float(X[i] @ X[j] / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-12))
           for i, j in itertools.combinations(range(len(X)), 2)]
    C = X - X.mean(0, keepdims=True)
    cen = [float(C[i] @ C[j] / (np.linalg.norm(C[i]) * np.linalg.norm(C[j]) + 1e-12))
           for i, j in itertools.combinations(range(len(C)), 2)]
    out = {"raw": float(np.mean(raw)), "centered": float(np.mean(cen)), "n": len(X)}
    if Y is not None and len(set(Y)) > 1:
        N = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = N @ N.T; np.fill_diagonal(S, -9)
        Y = np.array(Y)
        acc = float((Y[S.argmax(1)] == Y).mean())
        cnt = collections.Counter(Y); n = len(Y)
        ch = float(np.mean([(cnt[y] - 1) / (n - 1) for y in Y]))
        out |= {"probe_acc": acc, "chance": ch, "ratio": acc / ch if ch else None}
    print(f"  {name:<12} raw {out['raw']:.4f} · centered {out['centered']:+.4f}"
          + (f" · 잔차 장면검색 {out['probe_acc']:.3f} (우연 {out['chance']:.3f}, {out['ratio']:.1f}배)"
             if "probe_acc" in out else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--scene-only", action="store_true", help="장면 라벨이 있는 클립만")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _ev.enforce_determinism(0)
    chat = _ev.build_chat(a.cfg, a.ckpt, a.gpu_id, use_background=True)
    m, dev = chat.model, f"cuda:{a.gpu_id}"

    anno = json.load(open(a.anno)); vids, seen = [], set()
    for r in (anno if isinstance(anno, list) else anno.get("annotations", [])):
        v = r.get("video") or ""
        p = v if os.path.isabs(v) else os.path.join(_ev.ROOT, "hawk_anomaly/Videos", v)
        if v in seen or not os.path.exists(p) or os.path.getsize(p) >= 2 * 2**30: continue
        if a.scene_only and not scene_of(p): continue
        seen.add(v); vids.append(p)
        if len(vids) >= a.n: break

    BG, MO, AP, Y = [], [], [], []
    for i, p in enumerate(vids, 1):
        try:
            v, vm, vb = load_streams_aligned(p, n_frms=32, image_size=224, sampling="uniform", ablation="flow")
            v, vm, vb = apply_shared_transform(chat.vis_processor.transform, (v, vm, vb))
            with torch.no_grad():
                eb, _, _ = m.encode_videoQformer_visual(vb.unsqueeze(0).to(dev), background=True)
                em, _, _ = m.encode_videoQformer_visual(vm.unsqueeze(0).to(dev), motion=True)
                ea, _, _ = m.encode_videoQformer_visual(v.unsqueeze(0).to(dev))
            BG.append(eb.float().cpu().numpy().ravel()); MO.append(em.float().cpu().numpy().ravel())
            AP.append(ea.float().cpu().numpy().ravel()); Y.append(scene_of(p))
        except Exception as e:
            print(f"  [실패] {os.path.basename(p)}: {e}")
        if i % 20 == 0: print(f"  {i}/{len(vids)}", flush=True)

    Y = Y if all(Y) else None
    res = {k: stats(k, np.array(X), Y) for k, X in
           [("배경", BG), ("움직임", MO), ("외형", AP)]}
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
