#!/usr/bin/env python
"""완결된 epoch만으로 손실 추세를 본다 (부분 집계 제외)."""
import re, sys, collections, statistics as st
MIN = 200
log = sys.argv[1] if len(sys.argv) > 1 else '/home/work/seoik/runs/abl_flow/train.log'
ori = collections.defaultdict(list); mid = collections.defaultdict(list)
for l in open(log, errors='ignore'):
    m = re.search(r'Train: data epoch: \[(\d+)\].*?oriloss: (nan|[0-9.]+).*?middleloss: (nan|[0-9.]+)', l)
    if m:
        e = int(m.group(1))
        if m.group(2) != 'nan': ori[e].append(float(m.group(2)))
        if m.group(3) != 'nan': mid[e].append(float(m.group(3)))
done = [e for e in sorted(ori) if len(ori[e]) >= MIN]
part = [e for e in sorted(ori) if len(ori[e]) < MIN]
print(f"완결 epoch {len(done)}개 (iteration >= {MIN}), 부분 집계 제외: {part}")
print("epoch  oriloss   middleloss")
prev = None
for e in done[-10:]:
    o = st.mean(ori[e]); mv = st.mean(mid[e]) if mid[e] else float('nan')
    ar = "" if prev is None else ("↑" if o > prev else "↓")
    print(f"  {e:>2}   {o:.4f} {ar}   {mv:.4f}")
    prev = o
d = [st.mean(ori[e]) for e in done[-6:]]
run = mx = 0
for i in range(1, len(d)):
    run = run + 1 if d[i] > d[i-1] else 0; mx = max(mx, run)
print(f"\n최근 6 완결 epoch 최대 연속 상승 = {mx}회 (개입 기준 3회)")
