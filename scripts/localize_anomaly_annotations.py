#!/usr/bin/env python
"""
Stage-2 이상행동 annotation의 비디오 경로를 이 서버 배치에 맞게 로컬화한다.

왜 필요한가
-----------
원본 HAWK가 배포한 annotation JSON의 `video` 필드는 **원저자 서버의 절대경로**다:

    /remote-home/share/jiaqitang/Data/DoTA/Video/xxx.mp4

그런데 로더는 다음과 같이 조합한다 (video_instruct_dataset.py:127-128):

    rel_video_fp  = sample['video']
    full_video_fp = os.path.join(self.vis_root, rel_video_fp)

`os.path.join`은 두 번째 인자가 절대경로면 **첫 인자를 통째로 버린다.** 따라서
config의 `videos_dir`를 아무리 고쳐도 무시되고 `/remote-home/...`를 열려다 실패한다.
→ `video` 필드를 **videos_dir 기준 상대경로**로 바꿔야 한다.

게다가 annotation의 데이터셋 이름과 실제 배포본의 디렉토리 이름이 다르다
(`UCF-Crime` vs `UCF_Crime`, `avenue/avenue/...` vs `CUHK_Avenue/...` 등).
단순 prefix 치환이 아니라 아래 PATH_MAP이 필요한 이유다.

사용법
------
    python scripts/localize_anomaly_annotations.py            # 기본 경로로 전체 변환
    python scripts/localize_anomaly_annotations.py --check    # 쓰지 않고 검증만

원본 JSON은 건드리지 않고 `*.local.json`을 새로 쓴다.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
ANNO_DIR = os.path.join(ROOT, "hawk_anomaly", "Annotation", "All_Mix")
VIDEOS_DIR = os.path.join(ROOT, "hawk_anomaly", "Videos")

# 원저자 절대경로 prefix (annotation에 박혀 있는 값)
SRC_PREFIX = "/remote-home/share/jiaqitang/Data/"

# annotation 상대경로 → 실제 배포본 상대경로.
# 첫 매치가 적용되므로 더 구체적인 규칙을 앞에 둔다.
PATH_MAP = [
    (r"^DoTA/",                 "DoTA/"),           # 그대로
    (r"^UBnormal/",             "UBnormal/"),       # 그대로
    (r"^UCF-Crime/",            "UCF_Crime/"),      # 하이픈 → 언더스코어
    (r"^ShanghaiTechDataset/",  "ShanghaiTech/"),   # 이름 축약
    (r"^avenue/avenue/",        "CUHK_Avenue/"),    # 중복 레벨 제거 + 정식명
    (r"^ped1/ped1/",            "Ped1/"),           # 중복 레벨 제거 + 대문자
    (r"^ped2/ped2/",            "Ped2/"),
]


def build_index(videos_dir):
    """Videos/ 를 한 번만 순회해서 상대경로 집합을 만든다.

    파일마다 os.path.exists를 호출하면 NFS 왕복(파일당 ~35ms)이 7천 번 발생한다.
    한 번 walk 해서 set으로 들고 비교하는 편이 훨씬 빠르다.
    """
    index = set()
    for dirpath, _, filenames in os.walk(videos_dir):
        rel_dir = os.path.relpath(dirpath, videos_dir)
        for fn in filenames:
            index.add(fn if rel_dir == "." else os.path.join(rel_dir, fn))
    return index


def localize(video_path):
    """원저자 절대경로 → videos_dir 기준 상대경로. 매핑 실패 시 None."""
    if not video_path.startswith(SRC_PREFIX):
        return None
    rel = video_path[len(SRC_PREFIX):]
    for pattern, replacement in PATH_MAP:
        if re.match(pattern, rel):
            return re.sub(pattern, replacement, rel, count=1)
    return None


def process(src, dst, index, write):
    with open(src) as f:
        samples = json.load(f)

    ok, unmapped, missing = 0, [], []
    for s in samples:
        rel = localize(s["video"])
        if rel is None:
            unmapped.append(s["video"])
            continue
        if rel not in index:
            missing.append(rel)
            continue
        s["video"] = rel
        ok += 1

    name = os.path.basename(src)
    print(f"\n[{name}] 총 {len(samples)}개")
    print(f"  변환 성공     : {ok}")
    print(f"  매핑 규칙 없음: {len(unmapped)}")
    print(f"  파일 없음     : {len(missing)}")

    for label, items in (("매핑 실패", unmapped), ("파일 없음", missing)):
        if items:
            head = Counter(i.split("/")[0] for i in items).most_common(5)
            print(f"  └ {label} 상위: {head}")
            for i in items[:3]:
                print(f"     예) {i}")

    if write and not unmapped and not missing:
        with open(dst, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        print(f"  → 기록: {dst}")
    elif write:
        print("  → 불일치가 있어 기록하지 않음 (원본 보존)")

    return not unmapped and not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", default=ANNO_DIR)
    ap.add_argument("--videos-dir", default=VIDEOS_DIR)
    ap.add_argument("--check", action="store_true", help="쓰지 않고 검증만")
    args = ap.parse_args()

    print(f"videos_dir 인덱싱: {args.videos_dir}")
    index = build_index(args.videos_dir)
    print(f"  파일 {len(index)}개 발견")

    all_ok = True
    for split in ("train", "test", "all"):
        src = os.path.join(args.anno_dir, f"all_videos_{split}.json")
        if not os.path.exists(src):
            continue
        dst = os.path.join(args.anno_dir, f"all_videos_{split}.local.json")
        all_ok &= process(src, dst, index, write=not args.check)

    print("\n검증 통과" if all_ok else "\n불일치 있음 — 위 내역 확인 필요")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
