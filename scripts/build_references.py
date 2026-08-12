#!/usr/bin/env python
"""투고용 참고문헌 목록을 조립하고 인용 무결성을 검사한다.

왜 필요한가
-----------
원고는 원본 HAWK 논문의 번호 [1]–[49]를 승계하고 CERBERUS가 새로 인용하는 문헌에
[50]부터 이어 붙이는 방식으로 작성되어 있다. 초안 단계에서는 이 규칙이 편하지만,
투고 시점에는 두 가지가 필요하다.

1. **실제로 인용된 것만** 목록에 남아야 한다. 승계 번호 49개 중 본문이 인용하는 것은
   26개뿐이며, 나머지를 그대로 실으면 심사자가 "인용하지 않은 문헌을 채웠다"고 읽는다.
2. **번호가 연속**이어야 한다. 지금은 5, 7, 9, 10, … 처럼 구멍이 뚫려 있다.

이 스크립트는 본문을 훑어 실제 인용을 수집하고, 등장 순서대로 1부터 다시 매긴 매핑과
조립된 목록을 낸다. `--apply` 없이는 원고를 건드리지 않는다.

검사 항목
---------
- 인용됐으나 문헌 항목이 없는 번호 (dangling citation)
- 항목만 있고 인용되지 않은 번호 (uncited entry)
- 범위 표기(`[0,255]`, `[0,1]`)를 인용으로 오인하지 않는지

사용
----
    $CERBERUS_PY scripts/build_references.py                 # 검사 + 매핑 출력
    $CERBERUS_PY scripts/build_references.py --out refs.md   # 조립된 목록 저장
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
PAPER = f"{ROOT}/hawk/paper_translation"
BODY_GLOB = f"{PAPER}/improved/0[0-6]*.md"
APPENDIX = f"{PAPER}/improved/08_appendix.md"
ORIGIN_REFS = f"{PAPER}/origin/07_references.md"
NEW_REFS = f"{PAPER}/improved/07_references.md"

# 인용으로 볼 최대 번호. 이보다 크면 범위·좌표 표기로 간주한다.
MAX_REF = 200


def parse_entries(path):
    """`[n] 저자 …` 형식의 문헌 항목을 {번호: 본문} 으로 읽는다."""
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if m:
            if cur is not None:
                out[cur] = " ".join(buf).strip()
            cur, buf = int(m.group(1)), [m.group(2)]
        elif cur is not None:
            if line.strip().startswith(("#", ">")) or not line.strip():
                out[cur] = " ".join(buf).strip()
                cur, buf = None, []
            else:
                buf.append(line.strip())
    if cur is not None:
        out[cur] = " ".join(buf).strip()
    return out


def collect_citations(files):
    """등장 순서를 보존하며 인용 번호를 모은다.

    `[0, 255]` 같은 범위 표기를 걸러낸다 — 0 은 문헌 번호가 될 수 없고,
    MAX_REF 를 넘는 값이 섞인 묶음도 좌표·범위로 본다.
    """
    order, seen = [], set()
    for f in files:
        for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", open(f).read()):
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            if any(n == 0 or n > MAX_REF for n in nums):
                continue
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    order.append(n)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="조립된 목록을 쓸 경로 (없으면 검사만)")
    args = ap.parse_args()

    files = sorted(glob.glob(BODY_GLOB)) + [APPENDIX]
    files = [f for f in files if os.path.exists(f)]
    cited = collect_citations(files)

    entries = parse_entries(ORIGIN_REFS)
    entries.update(parse_entries(NEW_REFS))   # 신규가 승계를 덮어쓴다

    dangling = [n for n in cited if n not in entries]
    uncited = sorted(set(entries) - set(cited))

    print(f"본문 {len(files)}개 파일에서 인용 {len(cited)}건")
    print(f"  승계([1]–[49]) {sum(1 for n in cited if n < 50)}건 · "
          f"신규([50]–) {sum(1 for n in cited if n >= 50)}건")
    print(f"  문헌 항목 보유 {len(entries)}건")
    print(f"\n인용됐으나 항목 없음: {dangling or '없음'}")
    print(f"항목만 있고 미인용: {len(uncited)}건 "
          f"(투고 목록에서 제외됨: {uncited[:12]}{' …' if len(uncited) > 12 else ''})")

    if dangling:
        print("\n⚠ dangling citation 이 있으면 투고 목록을 만들 수 없습니다.")
        return 1

    print(f"\n=== 재번호 매핑 (등장 순서, 총 {len(cited)}개) ===")
    mapping = {old: i + 1 for i, old in enumerate(cited)}
    for old in cited[:10]:
        print(f"  [{old}] → [{mapping[old]}]  {entries[old][:60]}…")
    if len(cited) > 10:
        print(f"  … 외 {len(cited) - 10}건")

    if args.out:
        lines = ["# References (투고용 — 자동 조립)\n",
                 f"본문 등장 순서로 재번호하였다. 총 {len(cited)}건.",
                 "인용되지 않은 승계 항목은 제외하였다.\n"]
        for old in cited:
            lines.append(f"[{mapping[old]}] {entries[old]}\n")
        with open(args.out, "w") as f:
            f.write("\n".join(lines))
        print(f"\n기록: {args.out}")
        print("※ 원고의 인용 번호는 바꾸지 않았습니다. 본문 치환은 투고 직전에 "
              "한 번만 수행하십시오 — 중간에 하면 이후 편집에서 다시 어긋납니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
