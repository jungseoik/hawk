"""
Extract jxie/webvid_10m parquet (video bytes + text) into the on-disk layout that
HAWK's existing WebvidDataset expects, so NO model/dataset code changes are needed.

Output layout (matches WebvidDataset._get_video_path + 'name' caption column):
    <out>/videos/<page_dir>/<videoid>.mp4
    <out>/annotations/<page_dir>.csv     # columns: page_dir, videoid, name

page_dir = <prefix> + <shard number>, e.g. "m00000". The prefix must be a NON-numeric
string: a purely-numeric "00000" is parsed back as int 0 by pandas, which breaks
WebvidDataset._get_video_path and sends __getitem__ into an infinite resample loop.
A per-repo prefix (m/a/b...) also keeps page_dir unique across repos whose shard
numbers collide (main vs part_0 vs part_1 all start at train-00000).

Byte copy only (no re-encode) -> fast; random training reads become 1 file/video.

CLI usage (single repo dir):
    python scripts/extract_webvid_parquet.py \
        --parquet_dir data/webvid_10m/data --out_dir data/webvid_extracted \
        --prefix m [--limit N]

Importable: extract_shard(shard_path, out_dir, page_dir, limit=0) -> int
"""
import argparse, glob, os, csv
import pyarrow.parquet as pq


def extract_shard(shard_path: str, out_dir: str, page_dir: str, limit: int = 0) -> int:
    """Extract one parquet shard to <out_dir>/videos/<page_dir>/*.mp4 + CSV. Returns count."""
    vdir = os.path.join(out_dir, "videos", page_dir)
    ann_root = os.path.join(out_dir, "annotations")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(ann_root, exist_ok=True)
    rows_csv = os.path.join(ann_root, f"{page_dir}.csv")
    pf = pq.ParquetFile(shard_path)
    n = 0
    with open(rows_csv, "w", newline="") as cf:
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
                if limit and n >= limit:
                    return n
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True, help="dir with train-*.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="p", help="non-numeric page_dir prefix (e.g. m/a/b)")
    ap.add_argument("--limit", type=int, default=0, help="max videos total (0 = all)")
    args = ap.parse_args()
    assert not args.prefix[:1].isdigit(), "prefix must start non-numeric"

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert shards, f"no parquet in {args.parquet_dir}"
    total = 0
    for shard in shards:
        num = os.path.basename(shard).split("-")[1]  # "00000"
        page_dir = f"{args.prefix}{num}"
        rem = (args.limit - total) if args.limit else 0
        if args.limit and rem <= 0:
            break
        c = extract_shard(shard, args.out_dir, page_dir, limit=rem)
        total += c
        print(f"[extract] {os.path.basename(shard)} -> {page_dir} ({c} videos, total {total})", flush=True)
    print(f"[extract] DONE. {total} videos -> {args.out_dir}/videos", flush=True)


if __name__ == "__main__":
    main()
