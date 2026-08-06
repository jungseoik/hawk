"""
Download a CERBERUS checkpoint (and run metadata) from the Hugging Face model repo
into a local run dir, so training can be RESUMED on another server.

Counterpart to upload_checkpoint.py. Token resolution (in order):
--token, $HF_TOKEN, /data/pia/.hf_token, ~/.hf_token, cached hf login.

Usage:
  # pull the newest checkpoint into a run dir laid out for train_run.sh auto-resume
  python scripts/download_checkpoint.py --run-dir /data/<you>/runs/core --latest
  # or a specific one
  python scripts/download_checkpoint.py --run-dir /data/<you>/runs/core --epoch 53
After download, `bash scripts/train_run.sh <cfg> core <gpus> <nproc>` auto-resumes
from the newest checkpoint_*.pth found under <run-dir>/main/.
"""
import argparse, os, sys


def resolve_token(explicit=None):
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in ("/data/pia/.hf_token", os.path.expanduser("~/.hf_token")):
        if os.path.isfile(p):
            return open(p).read().strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="backseollgi/Cerberus")
    ap.add_argument("--folder", default="stage1_core", help="repo subfolder")
    ap.add_argument("--run-dir", required=True, help="local run dir; checkpoint lands in <run-dir>/main/")
    ap.add_argument("--epoch", type=int, default=None, help="specific checkpoint epoch to fetch")
    ap.add_argument("--latest", action="store_true", help="fetch the newest checkpoint on HF")
    ap.add_argument("--meta", action="store_true", help="also fetch config.yaml / train.log / STOPPED.md")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=resolve_token(args.token))
    files = api.list_repo_files(args.repo, repo_type="model")
    cks = [f for f in files if f.startswith(f"{args.folder}/checkpoint_") and f.endswith(".pth")]
    if not cks:
        print(f"[download] no checkpoints under {args.repo}:{args.folder}/"); sys.exit(1)

    def epoch_of(f):
        return int("".join(filter(str.isdigit, os.path.basename(f))) or -1)

    if args.epoch is not None:
        target = f"{args.folder}/checkpoint_{args.epoch}.pth"
        if target not in cks:
            print(f"[download] {target} not on HF. available: {sorted(epoch_of(c) for c in cks)}"); sys.exit(1)
    else:  # default / --latest
        target = max(cks, key=epoch_of)

    dest_dir = os.path.join(args.run_dir, "main")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[download] {args.repo}:{target} -> {dest_dir}/", flush=True)
    hf_hub_download(repo_id=args.repo, repo_type="model", filename=target,
                    local_dir=args.run_dir, local_dir_use_symlinks=False)
    # hf_hub_download preserves the repo path; move it to <run-dir>/main/checkpoint_N.pth
    got = os.path.join(args.run_dir, target)
    final = os.path.join(dest_dir, os.path.basename(target))
    if os.path.abspath(got) != os.path.abspath(final):
        os.replace(got, final)
    print(f"[download] ready: {final}  (epoch {epoch_of(target)})")

    if args.meta:
        for name in ("config.yaml", "run_info.txt", "train.log", "STOPPED.md", "loss_curve.png"):
            rp = f"{args.folder}/{name}"
            if rp in files:
                hf_hub_download(repo_id=args.repo, repo_type="model", filename=rp,
                                local_dir=args.run_dir, local_dir_use_symlinks=False)
                src = os.path.join(args.run_dir, rp)
                dst = os.path.join(args.run_dir, name)
                if os.path.abspath(src) != os.path.abspath(dst):
                    os.replace(src, dst)
                print(f"[download] meta: {name}")
    print("[download] done — next: bash scripts/train_run.sh <cfg> "
          f"{os.path.basename(args.run_dir)} <gpus> <nproc>  (auto-resumes from epoch {epoch_of(target)})")


if __name__ == "__main__":
    main()
