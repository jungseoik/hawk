#!/usr/bin/env python
"""GPU 공회전 — 인스턴스가 미사용으로 회수되는 것을 막는다.

실제 작업(학습·평가·진단)이 돌면 **스스로 비켜난다.** 그래야 공회전이 본 작업의
메모리나 연산을 빼앗지 않는다. 확인 주기마다:

  실제 작업 있음  → GPU 를 놓고(할당 해제) 대기만 한다
  실제 작업 없음  → 각 GPU 에서 작은 행렬곱을 잠깐 돌린다

메모리는 의도적으로 작게 잡는다(기본 512x512 행렬). 공회전이 OOM 의 원인이 되면
안 된다 — 이 컨테이너에서 정체불명 사망의 원인은 늘 메모리였다.
"""
import argparse, os, subprocess, sys, time

BUSY_PATTERNS = ["train[.]py --cfg-path", "evaluate[.]py", "diag_branch_sensitivity[.]py",
                 "places_probe[.]py"]


def busy():
    for p in BUSY_PATTERNS:
        try:
            if subprocess.run(["pgrep", "-f", p], capture_output=True).stdout.strip():
                return True
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--size", type=int, default=512, help="행렬 한 변 (작게 유지할 것)")
    ap.add_argument("--work-sec", type=float, default=5.0, help="한 주기에 도는 시간")
    ap.add_argument("--idle-sec", type=float, default=25.0, help="주기 간 대기")
    a = ap.parse_args()

    import torch
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    print(f"[keepalive] GPU {gpus} · {a.size}x{a.size} · {a.work_sec}s 작업 / {a.idle_sec}s 대기",
          flush=True)
    state = None
    while True:
        b = busy()
        if b != state:
            print(f"[keepalive] {'본 작업 감지 — 비켜남' if b else '유휴 — 공회전 시작'} "
                  f"{time.strftime('%F %T')}", flush=True)
            state = b
        if b:
            torch.cuda.empty_cache()
            time.sleep(a.idle_sec)
            continue
        try:
            for g in gpus:
                dev = f"cuda:{g}"
                x = torch.randn(a.size, a.size, device=dev)
                t0 = time.time()
                while time.time() - t0 < a.work_sec / len(gpus):
                    x = torch.mm(x, x)
                    x = x / (x.norm() + 1e-6)   # 발산 방지
                del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            # 본 작업이 방금 GPU 를 다 잡았을 수 있다. 조용히 물러난다.
            print(f"[keepalive] GPU 사용 불가, 대기: {str(e)[:80]}", flush=True)
            time.sleep(a.idle_sec)
        time.sleep(a.idle_sec)


if __name__ == "__main__":
    main()
