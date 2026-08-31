#!/usr/bin/env python
"""장면 식별 — "설명만 보고 어느 장소인지 알 수 있는가".

왜 이 지표인가
--------------
지금까지 배경 이해의 근거는 Scene-word Recall 이었는데, 그것은 **우리가 만든** 어휘
사전 기반 지표이고 정답도 HAWK 의 이상행동 설명문(배경 언급은 부수적)이다.
"당신들 지표에서만 좋은 것 아니냐"에 답이 안 된다.

여기서는 **데이터셋이 이미 갖고 있는 객관적 라벨**을 쓴다.
  ShanghaiTech: `01_0014_video.mp4` → 장면 ID `01` (캠퍼스 13개 장소)
  UBnormal:     `Scene1/…`          → Scene1, Scene2, …
LLM 생성 정답도, 사람 라벨링도 필요 없다.

과제: 각 클립의 **모델 생성 설명**을 벡터로 만들고, 가장 비슷한 다른 클립의 설명을 찾는다.
그 클립이 같은 장소면 정답. 배경을 잘 기술하는 모델일수록 같은 장소끼리 뭉친다.
설명에 장소 정보가 없으면 무작위에 가까워진다.

주의: 우연 수준(chance)은 장면별 클립 수에 따라 달라지므로 함께 보고한다.
"""
import argparse, json, os, re, math, random, collections

STOP = set("""a an the of in on at to and or is are was were be been being this that these those
it its as by for with from he she they them his her their there here what which who whom when
where why how video shows showing depicts depicting scene appears seems possible could would
also very some any all no not into over under after before during while""".split())


def scene_of(path):
    m = re.search(r"ShanghaiTech/.*?/(\d\d)_\d+", path)
    if m: return "ST:" + m.group(1)
    m = re.search(r"UBnormal/(Scene\d+)/", path)
    if m: return "UB:" + m.group(1)
    return None


def toks(t):
    return [w for w in re.findall(r"[a-z]+", (t or "").lower()) if w not in STOP and len(w) > 2]


def tfidf(docs):
    df = collections.Counter()
    for d in docs: df.update(set(d))
    N = len(docs)
    out = []
    for d in docs:
        tf = collections.Counter(d)
        v = {w: (1 + math.log(c)) * math.log(N / (1 + df[w])) for w, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({w: x / n for w, x in v.items()})
    return out


def cos(a, b):
    if len(a) > len(b): a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def retrieval(records):
    """각 클립의 최근접 이웃이 같은 장소인가. 장면당 2개 이상인 클립만 대상."""
    items = [(r["video"], scene_of(r["video"]), r.get("pred_description"))
             for r in records if scene_of(r["video"])]
    cnt = collections.Counter(s for _, s, _ in items)
    items = [(v, s, d) for v, s, d in items if cnt[s] >= 2]
    if len(items) < 4: return None
    vecs = tfidf([toks(d) for _, _, d in items])
    hits, per_clip = 0, {}
    for i in range(len(items)):
        best, bj = -1.0, None
        for j in range(len(items)):
            if i == j: continue
            c = cos(vecs[i], vecs[j])
            if c > best: best, bj = c, j
        ok = int(items[bj][1] == items[i][1])
        hits += ok
        per_clip[items[i][0]] = ok
    # 우연 수준: 같은 장면의 다른 클립 비율의 평균
    n = len(items)
    chance = sum((cnt[s] - 1) / (n - 1) for _, s, _ in items) / n
    return {"acc": hits / n, "n": n, "chance": chance,
            "n_scenes": len(set(s for _, s, _ in items)), "per_clip": per_clip}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for p in a.eval:
        d = json.load(open(p))
        r = retrieval(d["records"])
        name = re.sub(r"^eval_|\.json$", "", os.path.basename(p))
        res[name] = r
        if r:
            print(f"  {name:<22} 장면식별 {r['acc']:.3f}  (우연 {r['chance']:.3f}, "
                  f"n={r['n']}, 장면 {r['n_scenes']}종, 우연대비 {r['acc']/r['chance']:.2f}배)")
        else:
            print(f"  {name:<22} 표본 부족")
    if a.out:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_clip"}
                   for k, v in res.items() if v}, open(a.out, "w"), indent=2)
