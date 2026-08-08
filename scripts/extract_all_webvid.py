"""
Extract ALL downloaded WebVid parquet shards into the single on-disk layout that
stage1_main.yaml points at (/home/work/seoik/webvid_extracted).

Unlike scripts/build_webvid_split.py (which split across a big+small disk on the
old server), this server has one 9.1T NFS volume mounted at /home/work/seoik, so
everything lands in one place and no union/symlink step is needed.

Resumable + parallel:
  - one worker per shard, N workers in flight
  - a shard counts as done only when <out>/annotations/<page_dir>.csv exists AND
    <out>/.done/<page_dir> marker exists (CSV is written to .tmp then renamed, so a
    killed worker never leaves a shard looking complete)
  - re-running skips finished shards

Usage:
    python scripts/extract_all_webvid.py [--workers 8] [--out /home/work/seoik/webvid_extracted]
"""
import argparse, glob, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/home/work/seoik"
REPOS = [
    (f"{ROOT}/webvid_10m/data", "m"),
    (f"{ROOT}/webvid_10m_part_0/data", "a"),
    (f"{ROOT}/webvid_10m_part_1/data", "b"),
]
MIN_FREE_GB = 300  # stop extracting if the volume drops below this


def page_dir_of(shard, prefix):
    base = os.path.basename(shard)
    kind = "train" if base.startswith("train") else "val"
    num = base.split("-")[1]
    return f"{prefix}{num}" if kind == "train" else f"{prefix}v{num}"


def is_done(out_dir, page_dir):
    return os.path.exists(os.path.join(out_dir, ".done", page_dir))


def free_gb(path):
    u = shutil.disk_usage(path)
    return u.free / 1e9


def work(args):
    """Extract one shard. Writes CSV to .tmp then renames + drops a .done marker."""
    import csv
    import pyarrow.parquet as pq

    shard, out_dir, page_dir = args
    vdir = os.path.join(out_dir, "videos", page_dir)
    ann_root = os.path.join(out_dir, "annotations")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(ann_root, exist_ok=True)
    final_csv = os.path.join(ann_root, f"{page_dir}.csv")
    tmp_csv = final_csv + ".tmp"

    n = 0
    pf = pq.ParquetFile(shard)
    with open(tmp_csv, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["page_dir", "videoid", "name"])
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["video", "text"])
            for r in tbl.to_pylist():
                vid, txt = r.get("video"), r.get("text")
                b = vid.get("bytes") if isinstance(vid, dict) else None
                if not b or not txt:
                    continue
                with open(os.path.join(vdir, f"{n}.mp4"), "wb") as vf:
                    vf.write(b)
                w.writerow([page_dir, n, txt.replace("\n", " ").strip()])
                n += 1
    os.replace(tmp_csv, final_csv)
    os.makedirs(os.path.join(out_dir, ".done"), exist_ok=True)
    open(os.path.join(out_dir, ".done", page_dir), "w").close()
    return page_dir, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/webvid_extracted")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs, skipped = [], 0
    for d, prefix in REPOS:
        for shard in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            pd_ = page_dir_of(shard, prefix)
            if is_done(args.out, pd_):
                skipped += 1
                continue
            jobs.append((shard, args.out, pd_))

    print(f"[extract-all] shards: {len(jobs)} todo, {skipped} already done", flush=True)
    print(f"[extract-all] out={args.out}  free={free_gb(args.out):.0f} GB  workers={args.workers}", flush=True)
    if args.dry_run or not jobs:
        return

    t0 = time.time()
    total_vids, done = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j[2] for j in jobs}
        for fut in as_completed(futs):
            pd_ = futs[fut]
            try:
                _, n = fut.result()
            except Exception as e:  # keep going; the shard stays un-done and retries next run
                print(f"[extract-all] FAIL {pd_}: {type(e).__name__}: {e}", flush=True)
                continue
            done += 1
            total_vids += n
            el = time.time() - t0
            rate = done / el * 3600
            eta_h = (len(jobs) - done) / rate if rate else 0
            print(f"[extract-all] {pd_} -> {n} videos | {done}/{len(jobs)} shards | "
                  f"{total_vids} vids | {el/60:.1f} min | ETA {eta_h:.1f} h | free {free_gb(args.out):.0f} GB",
                  flush=True)
            if free_gb(args.out) < MIN_FREE_GB:
                print("[extract-all] STOP: disk below MIN_FREE_GB", flush=True)
                for f in futs:
                    f.cancel()
                break
    print(f"[extract-all] DONE {done} shards, {total_vids} videos in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
