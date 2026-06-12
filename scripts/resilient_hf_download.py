"""
Resilient Hugging Face dataset downloader with stall/slowdown detection + auto-restart.

WebVid mirror (jxie/webvid_10m*) sometimes throttles a resumed connection to a
crawl (~0.1 MB/s) and effectively hangs. The fix that works is: cancel, delete the
partial .incomplete, and relaunch on a fresh connection (hf_transfer). This watchdog
automates that for a multi-hour download.

Detection (per repo):
  - launch `hf download <repo> --include <pat> --local-dir <dir>` with HF_HUB_ENABLE_HF_TRANSFER=1
  - poll on-disk byte total every --poll s, compute MB/s
  - GRACE: ignore the initial hash-verify of already-present shards (no new bytes)
    until the first byte growth, but restart if nothing starts within --grace s
  - SLOWDOWN: after downloading starts, if throughput stays below --min-mbps for
    --stall s (covers both 0 MB/s hard stalls and 0.1 MB/s crawls), restart fresh
  - on hf exit 0 -> repo done; non-zero -> clean + relaunch
Restarts keep completed shards (hf skips them). Events are logged with timestamps.

Usage:
  python scripts/resilient_hf_download.py \
      --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
      --base /data/pia --include "data/*.parquet" \
      --stall 300 --poll 20 --min-mbps 3 --grace 900
"""
import argparse, glob, os, signal, subprocess, time

HF = os.path.expanduser("~/miniconda3/envs/cerberus/bin/hf")


def now():
    return time.strftime("%H:%M:%S")


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


def kill(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def fetch_repo(repo, include, d, stall, poll, min_mbps, grace):
    attempt = 0
    while True:
        attempt += 1
        p = launch(repo, include, d)
        t_launch = time.time()
        last = dir_bytes(d)
        started = False
        slow = 0.0
        print(f"[{now()}] {repo} attempt {attempt} pid={p.pid} (start {last/1e9:.1f} GB)", flush=True)
        restart = False
        while True:
            time.sleep(poll)
            rc = p.poll()
            cur = dir_bytes(d)
            mbps = (cur - last) / 1e6 / poll
            last = cur
            if rc is not None:
                if rc == 0:
                    print(f"[{now()}] {repo} DONE ({cur/1e9:.1f} GB)", flush=True)
                    return
                print(f"[{now()}] {repo} exited rc={rc}; clean+retry", flush=True)
                clean_partials(d)
                restart = True
                break
            if mbps > 0.5:
                started = True
            if not started:
                if time.time() - t_launch > grace:
                    print(f"[{now()}] {repo} GRACE>{grace}s no start -> restart", flush=True)
                    kill(p); clean_partials(d); restart = True; break
                continue
            if mbps < min_mbps:
                slow += poll
            else:
                slow = 0.0
            if slow >= stall:
                print(f"[{now()}] {repo} SLOW <{min_mbps}MB/s for {stall}s "
                      f"(now {mbps:.2f}MB/s, {cur/1e9:.1f} GB) -> kill+restart", flush=True)
                kill(p); clean_partials(d); restart = True; break
        if not restart:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--include", default="data/*.parquet")
    ap.add_argument("--stall", type=int, default=300, help="seconds below min-mbps => restart")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--min-mbps", type=float, default=3.0, help="below this MB/s counts as slow")
    ap.add_argument("--grace", type=int, default=900, help="max seconds for initial verify before forcing restart")
    args = ap.parse_args()
    t0 = time.time()
    for repo in args.repos:
        d = os.path.join(args.base, repo.split("/")[-1])
        fetch_repo(repo, args.include, d, args.stall, args.poll, args.min_mbps, args.grace)
    print(f"[{now()}] ALL DONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
