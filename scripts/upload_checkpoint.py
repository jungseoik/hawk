"""
Upload a training checkpoint (and its run metadata) to a Hugging Face model repo,
organized into per-run folders. Used to preserve milestone checkpoints of the
chunked CERBERUS training off the local box.

Token resolution (in order): --token, $HF_TOKEN, /data/pia/.hf_token, cached hf login.
Repo default: backseollgi/Cerberus. Skips files already present (idempotent).

Usage:
  python scripts/upload_checkpoint.py --ckpt <path> --folder stage1_core
  python scripts/upload_checkpoint.py --run-dir /data/pia/runs/core --folder stage1_core --latest
"""
import argparse, glob, os, sys


def resolve_token(explicit=None):
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in ("/data/pia/.hf_token", os.path.expanduser("~/.hf_token")):
        if os.path.isfile(p):
            return open(p).read().strip()
    return None  # fall back to cached hf login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="backseollgi/Cerberus")
    ap.add_argument("--folder", required=True, help="repo subfolder, e.g. stage1_core")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint file")
    ap.add_argument("--run-dir", default=None, help="run dir; used with --latest")
    ap.add_argument("--latest", action="store_true", help="pick newest checkpoint_*.pth under run-dir/main")
    ap.add_argument("--every", type=int, default=0, help="with --latest: only upload if epoch %% N == 0 (milestones)")
    ap.add_argument("--also", nargs="*", default=[], help="extra files to upload (run_info, config, log)")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=resolve_token(args.token))
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)

    ckpt = args.ckpt
    if args.latest and args.run_dir:
        cks = sorted(glob.glob(os.path.join(args.run_dir, "main", "checkpoint_*.pth")),
                     key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or -1))
        ckpt = cks[-1] if cks else None
    if not ckpt or not os.path.isfile(ckpt):
        print(f"[upload] no checkpoint to upload (ckpt={ckpt})"); sys.exit(0)

    if args.every and args.latest:
        n = int(''.join(filter(str.isdigit, os.path.basename(ckpt))) or -1)
        if n % args.every != 0:
            print(f"[upload] latest checkpoint epoch {n} not a multiple of {args.every} — skip (milestone-only)")
            sys.exit(0)

    existing = set(api.list_repo_files(args.repo, repo_type="model"))
    to_upload = [(ckpt, f"{args.folder}/{os.path.basename(ckpt)}")]
    # auto-attach run metadata if present next to run-dir
    if args.run_dir:
        for name in ("run_info.txt", "config.yaml", "train.log"):
            p = os.path.join(args.run_dir, name)
            if os.path.isfile(p):
                to_upload.append((p, f"{args.folder}/{name}"))
    for p in args.also:
        if os.path.isfile(p):
            to_upload.append((p, f"{args.folder}/{os.path.basename(p)}"))

    for local, remote in to_upload:
        # checkpoints are large + immutable per epoch → skip if already there; always refresh logs/meta
        if remote in existing and remote.endswith(".pth"):
            print(f"[upload] skip (exists): {remote}"); continue
        sz = os.path.getsize(local) / 1e9
        print(f"[upload] {local} -> {args.repo}:{remote} ({sz:.2f} GB)", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")
    print("[upload] done", flush=True)


if __name__ == "__main__":
    main()
