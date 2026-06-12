"""
Build the Stage-1 WebVid training set, split across a BIG disk and a SMALL disk,
then expose a single unified view via symlinks (no model/dataset code changes).

Routing: every Nth shard (default 5 -> 20%) goes to SMALL, the rest to BIG.
Per-repo page_dir prefix (m/a/b...) keeps page_dir unique across repos.
Hard capacity guard: never write to a disk already >= --cap (default 0.80) full.
Resumable: shards already extracted (videos/<page_dir>/ + annotations/<page_dir>.csv) are skipped.

After extraction, builds a union dir:
    <union>/videos/<page_dir>  -> symlink to whichever disk holds it
    <union>/annotations/*.csv  -> all per-shard CSVs gathered here
Point stage1 config: videos_dir=<union>/videos, anno_dir=<union>/annotations.

Usage (defaults match this server's layout):
    python scripts/build_webvid_split.py            # extract + union
    python scripts/build_webvid_split.py --dry-run  # show routing/capacity plan only
    python scripts/build_webvid_split.py --union-only
"""
import argparse, glob, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_webvid_parquet import extract_shard

DEFAULT_REPOS = [
    ("/data/pia/webvid_10m/data", "m"),
    ("/data/pia/webvid_10m_part_0/data", "a"),
    ("/data/pia/webvid_10m_part_1/data", "b"),
]
BIG = "/data/pia/webvid_extracted"
SMALL = "/home/pia/seoik/hawk/data/webvid_small"
UNION = "/data/pia/webvid_union"


def used_frac(path):
    p = path
    while not os.path.exists(p):
        p = os.path.dirname(p) or "/"
    u = shutil.disk_usage(p)
    return u.used / u.total


def enumerate_shards(repos):
    out = []
    for d, prefix in repos:
        for shard in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            num = os.path.basename(shard).split("-")[1]
            out.append((shard, prefix, num))
    return out


def already_done(dest, page_dir):
    return (os.path.isdir(os.path.join(dest, "videos", page_dir)) and
            os.path.exists(os.path.join(dest, "annotations", f"{page_dir}.csv")))


def build_union(big, small, union):
    vroot = os.path.join(union, "videos")
    aroot = os.path.join(union, "annotations")
    os.makedirs(vroot, exist_ok=True)
    os.makedirs(aroot, exist_ok=True)
    n_links = n_csv = 0
    for dest in (big, small):
        vd = os.path.join(dest, "videos")
        if os.path.isdir(vd):
            for pd in os.listdir(vd):
                link = os.path.join(vroot, pd)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(vd, pd), link)
                    n_links += 1
        ad = os.path.join(dest, "annotations")
        if os.path.isdir(ad):
            for csvf in glob.glob(os.path.join(ad, "*.csv")):
                tgt = os.path.join(aroot, os.path.basename(csvf))
                if not os.path.lexists(tgt):
                    os.symlink(csvf, tgt)
                    n_csv += 1
    print(f"[union] {n_links} page_dir links, {n_csv} csv links -> {union}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", default=BIG)
    ap.add_argument("--small", default=SMALL)
    ap.add_argument("--union", default=UNION)
    ap.add_argument("--small-every", type=int, default=5, help="every Nth shard -> small (5 = 20%)")
    ap.add_argument("--cap", type=float, default=0.80, help="never write to a disk >= this full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--union-only", action="store_true")
    ap.add_argument("--max-shards", type=int, default=0, help="process only first N shards (testing)")
    args = ap.parse_args()

    if args.union_only:
        build_union(args.big, args.small, args.union)
        return

    shards = enumerate_shards(DEFAULT_REPOS)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f"[plan] {len(shards)} shards | small=every {args.small_every} (~{100//args.small_every}%) | cap={args.cap:.0%}", flush=True)
    big_n = small_n = skip_n = done = 0
    for gidx, (shard, prefix, num) in enumerate(shards):
        dest = args.small if (gidx % args.small_every == 0) else args.big
        page_dir = f"{prefix}{num}"
        if already_done(dest, page_dir):
            skip_n += 1
            continue
        if args.dry_run:
            (small_n := small_n + 1) if dest == args.small else (big_n := big_n + 1)
            continue
        uf = used_frac(dest)
        if uf >= args.cap:
            print(f"[GUARD] {dest} at {uf:.0%} >= cap {args.cap:.0%} — skip {page_dir}", flush=True)
            skip_n += 1
            continue
        c = extract_shard(shard, dest, page_dir)
        done += c
        tag = "SMALL" if dest == args.small else "BIG"
        (small_n := small_n + 1) if dest == args.small else (big_n := big_n + 1)
        if (big_n + small_n) % 20 == 0:
            print(f"[extract] {tag} {page_dir} ({c}) | big={big_n} small={small_n} "
                  f"vids={done} | big_disk={used_frac(args.big):.0%} small_disk={used_frac(args.small):.0%}", flush=True)
    print(f"[plan] big_shards={big_n} small_shards={small_n} skipped={skip_n} extracted_videos={done}", flush=True)
    if not args.dry_run:
        build_union(args.big, args.small, args.union)
        print("[done] set videos_dir=%s/videos anno_dir=%s/annotations" % (args.union, args.union), flush=True)


if __name__ == "__main__":
    main()
