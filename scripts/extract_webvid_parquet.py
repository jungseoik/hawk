"""
Extract jxie/webvid_10m parquet (video bytes + text) into the on-disk layout that
HAWK's existing WebvidDataset expects, so NO model/dataset code changes are needed.

Output layout (matches WebvidDataset._get_video_path + 'name' caption column):
    <out>/videos/<page_dir>/<videoid>.mp4
    <out>/annotations/<page_dir>.csv     # columns: page_dir, videoid, name

WebvidDataset reads all *.csv under anno_root, concatenates, and loads
videos_dir/page_dir/videoid.mp4. We set page_dir = shard id (zero-padded), and
videoid = row index within the shard.

Byte copy only (no re-encode) -> fast. Random training reads become 1 file/video
instead of reading a ~450MB parquet row-group per sample.

Usage:
    conda run -n cerberus python scripts/extract_webvid_parquet.py \
        --parquet_dir data/webvid_10m/data --out_dir data/webvid_extracted [--limit N]
"""
import argparse, glob, os, csv
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True, help="dir with train-*.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max videos total (0 = all)")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert shards, f"no parquet in {args.parquet_dir}"
    vid_root = os.path.join(args.out_dir, "videos")
    ann_root = os.path.join(args.out_dir, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    total = 0
    for shard in shards:
        # Prefix with a letter so pandas keeps page_dir as a STRING when the
        # CSV is read back. A purely-numeric "00000" is parsed as int 0, which
        # breaks WebvidDataset._get_video_path (videos/0/.. != videos/p00000/..)
        # and sends __getitem__ into an infinite "file not found -> resample" loop.
        page_dir = "p" + os.path.basename(shard).split("-")[1]  # e.g. "p00000"
        vdir = os.path.join(vid_root, page_dir)
        os.makedirs(vdir, exist_ok=True)
        rows_csv = os.path.join(ann_root, f"{page_dir}.csv")
        pf = pq.ParquetFile(shard)
        n_shard = 0
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
                    vidid = n_shard
                    with open(os.path.join(vdir, f"{vidid}.mp4"), "wb") as vf:
                        vf.write(b)
                    w.writerow([page_dir, vidid, txt.replace("\n", " ").strip()])
                    n_shard += 1
                    total += 1
                    if args.limit and total >= args.limit:
                        break
                if args.limit and total >= args.limit:
                    break
        print(f"[extract] {os.path.basename(shard)} -> {n_shard} videos (total {total})", flush=True)
        if args.limit and total >= args.limit:
            break
    print(f"[extract] DONE. {total} videos -> {vid_root}\n"
          f"          annotations -> {ann_root}", flush=True)


if __name__ == "__main__":
    main()
