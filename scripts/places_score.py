#!/usr/bin/env python
"""Places365 프로브 채점 — 외부·객관 라벨 기준 배경 이해.

두 지표를 낸다.
  1) 최근접 이웃 검색 정확도: 어떤 이미지의 설명과 가장 비슷한 다른 설명이 같은 장면
     범주인가. 우연 수준은 범주별 표본 수로 계산해 함께 보고한다.
  2) 범주명 언급률: 생성된 설명이 정답 범주명(또는 그 구성 단어)을 담고 있는가.
     해석은 쉽지만 모델이 Places365 어휘를 쓸 이유가 없으므로 하한으로만 본다.

검정은 다른 모든 비교와 같은 **클립 단위 쌍 부트스트랩**을 쓴다.
"""
import argparse, json, math, os, re, random, collections, statistics as st

STOP = set("""a an the of in on at to and or is are was were be been being this that these those
it its as by for with from there here what which when where why how video image shows showing
depicts depicting scene appears seems likely possible could would also very some any all no not
into over under after before during while based given description provided location place""".split())


def toks(t):
    return [w for w in re.findall(r"[a-z]+", (t or "").lower()) if w not in STOP and len(w) > 2]


def tfidf(docs):
    df = collections.Counter()
    for d in docs: df.update(set(d))
    N = len(docs); out = []
    for d in docs:
        tf = collections.Counter(d)
        v = {w: (1 + math.log(c)) * math.log(N / (1 + df[w])) for w, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({w: x / n for w, x in v.items()})
    return out


def cos(a, b):
    if len(a) > len(b): a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def score(recs):
    recs = [r for r in recs if (r.get("pred") or "").strip()]
    vecs = tfidf([toks(r["pred"]) for r in recs])
    cnt = collections.Counter(r["cls"] for r in recs)
    n = len(recs)
    nn, mention = {}, {}
    for i, r in enumerate(recs):
        best, bj = -1.0, None
        for j in range(n):
            if i == j: continue
            c = cos(vecs[i], vecs[j])
            if c > best: best, bj = c, j
        nn[r["id"]] = int(recs[bj]["cls"] == r["cls"])
        # 범주명 구성 단어 중 하나라도 등장하면 언급으로 본다 ("ice cream parlor" → ice/cream/parlor)
        words = [w for w in re.split(r"[^a-z]+", r["name"].lower()) if len(w) > 2]
        low = (r["pred"] or "").lower()
        mention[r["id"]] = int(any(w in low for w in words))
    chance = sum((cnt[r["cls"]] - 1) / (n - 1) for r in recs) / n
    return {"n": n, "nn_acc": st.mean(nn.values()), "chance": chance,
            "mention": st.mean(mention.values()), "nn": nn, "men": mention}


def boot(a, b, B=10000, seed=0):
    ks = sorted(set(a) & set(b)); d = [a[k] - b[k] for k in ks]
    rnd = random.Random(seed); n = len(d)
    s = sorted(st.mean([d[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return st.mean(d), s[int(.025 * B)], s[int(.975 * B)], n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {}
    for p in a.probe:
        name = re.sub(r"^places_|\.json$", "", os.path.basename(p))
        R[name] = score(json.load(open(p))["records"])
    print(f"  {'모델':<16}{'장면검색':>10}{'우연':>8}{'배수':>7}{'범주명언급':>11}{'n':>6}")
    for k, v in R.items():
        print(f"  {k:<16}{v['nn_acc']:>10.3f}{v['chance']:>8.3f}{v['nn_acc']/v['chance']:>7.1f}"
              f"{v['mention']:>11.3f}{v['n']:>6d}")
    ks = list(R)
    print("\n  === 쌍 부트스트랩 (장면 검색 정확도) ===")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            o, lo, hi, n = boot(R[ks[i]]["nn"], R[ks[j]]["nn"])
            star = "  ***" if (lo > 0 or hi < 0) else ""
            print(f"    {ks[i]:<14} − {ks[j]:<14} Δ={o:+.4f} CI [{lo:+.4f}, {hi:+.4f}] n={n}{star}")
    print("\n  === 쌍 부트스트랩 (범주명 언급률) ===")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            o, lo, hi, n = boot(R[ks[i]]["men"], R[ks[j]]["men"])
            star = "  ***" if (lo > 0 or hi < 0) else ""
            print(f"    {ks[i]:<14} − {ks[j]:<14} Δ={o:+.4f} CI [{lo:+.4f}, {hi:+.4f}] n={n}{star}")
    if a.out:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("nn", "men")}
                   for k, v in R.items()}, open(a.out, "w"), indent=2)
