"""
Resilient Hugging Face dataset downloader with stall detection + auto-restart.

WebVid mirror (jxie/webvid_10m*) sometimes throttles a resumed connection to
~0.1 MB/s and effectively hangs. The fix that works is: cancel, delete the
partial .incomplete, and relaunch on a fresh connection (hf_transfer). This
watchdog automates exactly that so a multi-hour download self-heals unattended.

Behavior per repo:
  - launch `hf download <repo> --include <pat> --local-dir <dir>` with
    HF_HUB_ENABLE_HF_TRANSFER=1 (fast Rust downloader);
  - poll the on-disk byte total every --poll seconds;
  - if it does not grow for --stall seconds (and the process is still alive),
    treat it as stalled: kill the process group, delete *.incomplete/*.lock in
    the cache, and relaunch (HF resumes already-complete shards instantly);
  - if hf exits 0, that repo is done; if it exits non-zero, clean + relaunch.

Usage:
  python scripts/resilient_hf_download.py \
      --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
      --base /data/pia --include "data/*.parquet" --stall 120 --poll 20
"""
import argparse, glob, os, signal, subprocess, time

HF = os.path.expanduser("~/miniconda3/envs/cerberus/bin/hf")


def dir_bytes(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clean_partials(d):
    for pat in ("*.incomplete", "*.lock"):
        for p in glob.glob(f"{d}/.cache/**/{pat}", recursive=True):
            try:
                os.remove(p)
            except OSError:
                pass


def launch(repo, include, d):
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
    log = open(os.path.join(d, "_download.log"), "a")
    log.write(f"\n==== launch {repo} @ {int(time.time())} ====\n"); log.flush()
    hf = HF if os.path.exists(HF) else "hf"
    return subprocess.Popen(
        [hf, "download", repo, "--repo-type", "dataset", "--include", include,
         "--local-dir", d],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def fetch_repo(repo, include, d, stall, poll):
    attempt = 0
    while True:
        attempt += 1
        p = launch(repo, include, d)
        print(f"[{repo}] attempt {attempt} pid={p.pid} -> {d}", flush=True)
        last_bytes, last_change = dir_bytes(d), time.time()
        while True:
            time.sleep(poll)
            rc = p.poll()
            cur = dir_bytes(d)
            if cur > last_bytes:
                last_bytes, last_change = cur, time.time()
            if rc is not None:  # process ended
                if rc == 0:
                    print(f"[{repo}] DONE ({cur/1e9:.1f} GB)", flush=True)
                    return
                print(f"[{repo}] exited rc={rc}; clean+retry", flush=True)
                clean_partials(d)
                break
            if time.time() - last_change > stall:  # stalled
                gb = cur / 1e9
                print(f"[{repo}] STALL >{stall}s at {gb:.1f} GB -> kill+restart", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
                clean_partials(d)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="parent dir; each repo -> <base>/<repo_name>")
    ap.add_argument("--include", default="data/*.parquet")
    ap.add_argument("--stall", type=int, default=120, help="seconds of no growth => restart")
    ap.add_argument("--poll", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    for repo in args.repos:
        d = os.path.join(args.base, repo.split("/")[-1])
        fetch_repo(repo, args.include, d, args.stall, args.poll)
    print(f"ALL DONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
