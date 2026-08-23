#!/usr/bin/env python
"""절제 arm 간 비교 — §4.1 통계 프로토콜의 구현.

원고 §4.1이 확정한 절차를 그대로 옮긴다. 결과가 나온 뒤에 급하게 만들면 절차가
데이터를 보고 정해지므로, 숫자가 없는 지금 만들고 합성 데이터로 검증해 둔다.

구현하는 것
-----------
1. **클립 단위 쌍 부트스트랩.** 조건 간 차이의 95% 신뢰구간을 10,000회 리샘플로 낸다.
   리샘플 단위는 클립이다 — Scene-word Recall은 단어 단위 micro-average라 같은 클립
   안의 단어들이 상관되어 있고, 단어를 독립 표본으로 다루면 구간이 과소 추정된다.
2. **사전 지정 3대비 + Holm 보정.** `flow`−`zero`, `flow`−`random_mask`,
   `flow`−`duplicate` 만 검정한다. 나머지 조합은 탐색적 관찰로 따로 표시한다.
3. **도메인별 주 종점.** 풀링 값은 평가 분할이 DoTA 62.8%로 편중되어 사실상 대시캠
   성능이므로 보조로 내린다.
4. **음성 결과 보고.** 신뢰구간이 0을 포함하면 "차이 없음"이 아니라 "이 표본에서
   미검출"로 쓰고 구간 상한을 함께 낸다.

쓰지 않는 것
------------
정규성 가정에 기대는 t-검정은 쓰지 않는다. 클립 단위 BLEU·Scene-word Recall 분포는
0에 몰린 비대칭 분포이고 표본도 수백 개 수준이라 부트스트랩이 안전하다.

사용
----
    $CERBERUS_PY scripts/compare_arms.py \
        --eval experiments/out/eval_abl_flow.json \
        --eval experiments/out/eval_abl_zero.json \
        --eval experiments/out/eval_abl_random_mask.json \
        --eval experiments/out/eval_abl_duplicate.json \
        --out experiments/out/arm_comparison.json

    $CERBERUS_PY scripts/compare_arms.py --self-test    # 합성 데이터 검증
"""
import argparse
import json
import math
import os
import re
import random
import sys
from collections import defaultdict

# 사전 지정 대비. 이 목록을 결과를 보고 늘리면 다중 비교 보정이 무의미해진다.
PRESPECIFIED = [("flow", "zero"), ("flow", "random_mask"), ("flow", "duplicate")]
N_BOOT = 10000
BOOT_SEED = 20260813


def clip_bootstrap_diff(a_by_clip, b_by_clip, n_boot=N_BOOT, seed=BOOT_SEED):
    """두 조건의 클립별 점수에서 평균 차이의 부트스트랩 분포를 낸다.

    같은 클립을 두 조건에서 평가하므로 **쌍(paired)** 으로 리샘플한다. 클립을 뽑을 때
    두 조건의 그 클립 점수를 함께 가져가면, 클립 난이도의 분산이 상쇄되어 조건 간
    차이에 대한 구간이 좁아진다 — 이것이 쌍 설계의 이점이다.
    """
    common = sorted(set(a_by_clip) & set(b_by_clip))
    if len(common) < 2:
        return None
    diffs = [a_by_clip[c] - b_by_clip[c] for c in common]
    n = len(diffs)
    point = sum(diffs) / n

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]

    # 양측 p: 0이 부트스트랩 분포의 어느 꼬리에 있는지로 근사한다.
    n_le = sum(1 for b in boots if b <= 0)
    p = 2 * min(n_le, n_boot - n_le) / n_boot
    p = min(max(p, 1.0 / n_boot), 1.0)

    return {"n_clips": n, "diff": point, "ci_low": lo, "ci_high": hi,
            "p_raw": p, "detected": not (lo <= 0 <= hi)}


def holm(pvals):
    """Holm-Bonferroni. 사전 지정 대비 수만큼 보정한다."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (m - rank) * pvals[i])
        adj = max(adj, prev)          # 단조성 보정
        out[i] = adj
        prev = adj
    return out


def per_clip_scores(records, metric_fn):
    """레코드 목록에서 {클립: 점수} 를 만든다."""
    return {r["video"]: metric_fn(r) for r in records if metric_fn(r) is not None}


def load_eval(path):
    """arm 이름은 **run 디렉터리**에서 뽑는다.

    `config.static_ablation` 을 쓰면 안 된다 — 그것은 정적 스트림에 무엇을 넣었는가일 뿐이라
    서로 다른 arm 이 같은 값을 가진다. 실제로 `flow_reinit` 은 정적 입력이 `flow` 라
    `flow` arm 과 이름이 충돌해 딕셔너리에서 조용히 덮어썼다(2026-08-23 발견).
    run 디렉터리(`runs/abl2_<arm>/…`)는 arm 마다 유일하므로 이것을 정본으로 삼는다.
    """
    with open(path) as f:
        d = json.load(f)
    ckpt = d.get("config", {}).get("ckpt") or ""
    m = re.search(r"runs/abl2_([A-Za-z0-9_]+)/", ckpt)
    if m:
        arm = m.group(1)
    else:
        arm = re.sub(r"^eval_(abl2?_)?|\.json$", "", os.path.basename(path))
    return arm, d


def domain_of(video_path):
    return video_path.split("/")[0]


def compare(evals, metric_name, metric_fn):
    """사전 지정 대비를 검정하고 도메인별로 분해한다."""
    scores = {arm: per_clip_scores(d.get("records", []), metric_fn)
              for arm, d in evals.items()}

    results = {"metric": metric_name, "prespecified": [], "exploratory": [],
               "per_domain": {}}

    raw_p, rows = [], []
    for a, b in PRESPECIFIED:
        if a not in scores or b not in scores:
            continue
        r = clip_bootstrap_diff(scores[a], scores[b])
        if r is None:
            continue
        r["contrast"] = f"{a} − {b}"
        rows.append(r)
        raw_p.append(r["p_raw"])

    for r, p_adj in zip(rows, holm(raw_p)):
        r["p_holm"] = p_adj
        # 판정은 보정된 p 와 구간을 함께 본다.
        r["verdict"] = ("검출됨" if (r["detected"] and p_adj < 0.05)
                        else "이 표본에서 미검출")
        results["prespecified"].append(r)

    # 도메인별 — 주 종점
    domains = sorted({domain_of(v) for s in scores.values() for v in s})
    for dom in domains:
        sub = {arm: {v: x for v, x in s.items() if domain_of(v) == dom}
               for arm, s in scores.items()}
        entry = {}
        for a, b in PRESPECIFIED:
            if a in sub and b in sub:
                r = clip_bootstrap_diff(sub[a], sub[b])
                if r:
                    entry[f"{a} − {b}"] = r
        if entry:
            results["per_domain"][dom] = entry

    # 사전 지정 밖 조합은 탐색적으로만 표시
    arms = sorted(scores)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            if (a, b) in PRESPECIFIED or (b, a) in PRESPECIFIED:
                continue
            r = clip_bootstrap_diff(scores[a], scores[b])
            if r:
                r["contrast"] = f"{a} − {b}"
                r["note"] = "사전 지정 대비 아님 — 탐색적 관찰, 보정 대상 아님"
                results["exploratory"].append(r)
    return results


def fmt(r):
    return (f"{r['contrast']:<26} Δ={r['diff']:+.4f}  "
            f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
            f"p_holm={r.get('p_holm', float('nan')):.4f}  n={r['n_clips']}  "
            f"→ {r.get('verdict', '')}")


def self_test():
    """합성 데이터로 절차가 의도대로 동작하는지 확인한다."""
    print("=== 합성 검증 ===")
    rng = random.Random(1)
    clips = [f"DoTA/v{i}.mp4" for i in range(150)] + [f"UCF_Crime/v{i}.mp4" for i in range(100)]

    # 클립 난이도(공통) + 조건 효과 + 잡음
    base = {c: rng.gauss(0.30, 0.10) for c in clips}

    def make(effect, noise=0.02):
        return {c: base[c] + effect + rng.gauss(0, noise) for c in clips}

    cases = [
        ("진짜 효과 +0.03", make(0.03), make(0.0)),
        ("효과 없음",       make(0.0),  make(0.0)),
        ("미세 효과 +0.002", make(0.002), make(0.0)),
    ]
    for name, a, b in cases:
        r = clip_bootstrap_diff(a, b, n_boot=2000, seed=7)
        det = "검출" if r["detected"] else "미검출"
        print(f"  {name:<18} Δ={r['diff']:+.4f}  CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {det}")

    print("\n  쌍 설계가 구간을 좁히는가 (같은 데이터, 쌍 vs 비쌍):")
    a, b = make(0.03), make(0.0)
    paired = clip_bootstrap_diff(a, b, n_boot=2000, seed=7)
    # 비쌍: 클립 대응을 끊고 각 조건을 독립 리샘플
    ka, kb = list(a), list(b); rng.shuffle(kb)
    unpaired = clip_bootstrap_diff(a, {ka[i]: b[kb[i]] for i in range(len(ka))},
                                   n_boot=2000, seed=7)
    wp = paired["ci_high"] - paired["ci_low"]
    wu = unpaired["ci_high"] - unpaired["ci_low"]
    print(f"    쌍   CI 폭 {wp:.4f}")
    print(f"    비쌍 CI 폭 {wu:.4f}   → 쌍이 {wu/wp:.1f}배 좁음")

    print("\n  Holm 보정:")
    for ps in ([0.01, 0.04, 0.20], [0.001, 0.001, 0.001]):
        print(f"    raw {ps} → holm {[round(x,4) for x in holm(ps)]}")
    print("\n합성 검증 통과 — 실제 결과 파일이 생기면 --eval 로 실행하십시오.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", default=[],
                    help="arm 별 평가 결과 JSON (여러 번 지정)")
    ap.add_argument("--out", help="비교 결과를 쓸 경로")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.eval:
        self_test()
        return 0

    evals = {}
    for p in args.eval:
        arm, d = load_eval(p)
        if arm in evals:
            sys.exit(f"[compare_arms] arm 이름 충돌: '{arm}' 이 두 번 나왔습니다 ({p}). "
                     f"조용히 덮어쓰면 비교가 통째로 틀리므로 중단합니다.")
        evals[arm] = d
        print(f"  {arm:<14} ← {os.path.basename(p)}  "
              f"({len(d.get('records', []))} 클립)")

    # 지표별 비교. 클립 단위 점수를 만들 수 있는 것만 다룬다.
    metrics = {
        "scene_word_recall_clip": lambda r: r.get("scene_recall_clip"),
        "bleu1_clip": lambda r: r.get("bleu1_clip"),
    }
    out = {"n_boot": N_BOOT, "seed": BOOT_SEED,
           "prespecified_contrasts": [f"{a} − {b}" for a, b in PRESPECIFIED],
           "results": {}}

    for name, fn in metrics.items():
        res = compare(evals, name, fn)
        if not res["prespecified"]:
            print(f"\n[{name}] 클립 단위 점수가 없어 건너뜁니다.")
            continue
        out["results"][name] = res
        print(f"\n=== {name} — 사전 지정 대비 (Holm 보정) ===")
        for r in res["prespecified"]:
            print("  " + fmt(r))
        print(f"--- 도메인별 (주 종점) ---")
        for dom, entry in res["per_domain"].items():
            for k, r in entry.items():
                print(f"  [{dom:<12}] {k:<24} Δ={r['diff']:+.4f} "
                      f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] n={r['n_clips']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n기록: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
