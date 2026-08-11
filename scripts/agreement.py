#!/usr/bin/env python
"""사람 주석과 LLM 주석의 일치도(Cohen's κ)를 계산하고 불일치를 분해한다.

κ 하나만 보면 "얼마나 맞았나"는 알 수 있어도 **어디서 갈렸는가**를 모른다. 판정자
둘 중 어느 쪽이 옳은지는 κ가 답해주지 않으므로, 혼동 행렬과 불일치 사례를 함께 낸다.

특히 이 과제에서는 사람 쪽이 불리할 수 있다 — 동적 스트림은 시간 신호인데 사람에게는
정지 프레임 몇 장으로 제시되기 때문이다. 따라서 κ가 낮게 나오면 "LLM 이 틀렸다"가
아니라 "두 판정자가 서로 다른 정보를 봤다"일 수 있고, 그 판단에는 불일치 내역이 필요하다.

사용법:
    $CERBERUS_PY scripts/agreement.py
    $CERBERUS_PY scripts/agreement.py --show 15     # 불일치 사례 15건 출력
"""
import argparse
import json
import os
from collections import Counter

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
BASE = f"{ROOT}/hawk/experiments/bg_critical_benchmark"


def cohen_kappa(a, b, labels):
    """가중치 없는 Cohen's κ."""
    n = len(a)
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    po = sum(m[i][i] for i in range(len(labels))) / n
    row = [sum(r) for r in m]
    col = [sum(m[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum(row[i] * col[i] for i in range(len(labels))) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan"), po, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=f"{BASE}/labels_human.json")
    ap.add_argument("--llm", default=f"{BASE}/labels_llm.json")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    H = json.load(open(args.human))
    M = json.load(open(args.llm))
    common = [k for k in H if k in M]
    if not common:
        raise SystemExit("공통 클립이 없습니다.")

    h = [H[k]["motion_sufficiency"] for k in common]
    m = [M[k]["motion_sufficiency"] for k in common]
    labels = sorted(set(h) | set(m))

    kappa, po, mat = cohen_kappa(h, m, labels)
    print(f"공통 클립 {len(common)}건")
    print(f"단순 일치율 {po:.3f}   Cohen's κ = {kappa:.3f}")
    interp = ("거의 완전 일치" if kappa >= .81 else "상당한 일치" if kappa >= .61
              else "보통" if kappa >= .41 else "약함" if kappa >= .21 else "거의 없음")
    print(f"  → {interp}  (통상 기준: κ ≥ 0.6 이면 채택 가능)")

    print("\n=== 혼동 행렬 (행: 사람, 열: LLM) ===")
    w = max(len(l) for l in labels) + 1
    print(" " * w + "".join(f"{l[:9]:>10}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<{w}}" + "".join(f"{mat[i][j]:>10}" for j in range(len(labels))))

    print("\n=== 라벨별 분포 ===")
    ch, cm = Counter(h), Counter(m)
    for l in labels:
        print(f"  {l:<20} 사람 {ch.get(l,0):>4}   LLM {cm.get(l,0):>4}")

    # 우리가 실제로 쓰려는 것은 context_critical 집합이므로, 그 항목만 따로 본다.
    hc = {k for k in common if H[k]["motion_sufficiency"] == "context_critical"}
    mc = {k for k in common if M[k]["motion_sufficiency"] == "context_critical"}
    inter, union = hc & mc, hc | mc
    print(f"\n=== context_critical 집합 일치 ===")
    print(f"  사람 {len(hc)}건 · LLM {len(mc)}건 · 교집합 {len(inter)}건")
    if union:
        print(f"  Jaccard {len(inter)/len(union):.3f}")

    dis = [k for k in common if H[k]["motion_sufficiency"] != M[k]["motion_sufficiency"]]
    print(f"\n=== 불일치 {len(dis)}건 중 상위 {min(args.show, len(dis))}건 ===")
    for k in dis[: args.show]:
        hh, mm = H[k], M[k]
        print(f"  {k[:52]}")
        print(f"    사람: {hh['motion_sufficiency']:<18} (a1={hh.get('a_motion_only')}, a2={hh.get('a_original')})")
        print(f"    LLM : {mm['motion_sufficiency']:<18} (a1={mm.get('a_motion_only')}, a2={mm.get('a_original')})"
              f"  «{mm.get('what_original','')[:44]}»")


if __name__ == "__main__":
    main()
