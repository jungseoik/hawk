"""
E3 — Loss-direction Causal Verification
=======================================
Paper experiment E3 (docs/cerberus-research-plan.md §3).

The motion<->background dissimilarity loss must push the two complementary
representations APART. We compare two formulations under otherwise identical
conditions:

    buggy :  L = 1 - cos(z_m, z_b)        # minimizing => cos -> +1  (COLLAPSE)
    fixed :  L = (1 + cos(z_m, z_b)) / 2  # minimizing => cos -> -1  (DISENTANGLE)

This file provides:
  - the loss functions (torch),
  - a self-contained synthetic optimization that proves the convergence
    direction of each loss WITHOUT any dataset or trained model.

On the real model, the same losses are wired in `hawk/tasks/base_task.py`
(`mse_loss_bg`); this script is the controlled, isolated causal demonstration
that the paper's Ablation C cites.

Dependencies: torch (required for the demo); numpy.
"""
from __future__ import annotations

import argparse


def l_sim(cos: "float|object"):
    """Appearance<->motion similarity loss: 1 - cos. Minimizing pulls together."""
    return 1.0 - cos


def l_dis_buggy(cos):
    """BUGGY motion<->background loss: 1 - cos. Wrongly pulls together."""
    return 1.0 - cos


def l_dis_fixed(cos):
    """FIXED motion<->background loss: (1 + cos)/2. Correctly pushes apart."""
    return (1.0 + cos) / 2.0


def _run_demo(steps: int = 1500, dim: int = 256, lr: float = 0.5, seed: int = 0):
    import torch
    torch.manual_seed(seed)

    def optimize(loss_fn, label):
        # two free compact representations (as produced by the bottleneck projection)
        zm = torch.randn(1, dim, requires_grad=True)
        zb = torch.randn(1, dim, requires_grad=True)
        opt = torch.optim.SGD([zm, zb], lr=lr)
        traj = []
        for t in range(steps):
            opt.zero_grad()
            cos = torch.nn.functional.cosine_similarity(zm, zb).squeeze()
            loss = loss_fn(cos)
            loss.backward()
            opt.step()
            if t % (steps // 8) == 0 or t == steps - 1:
                traj.append((t, float(cos.detach())))
        final_cos = float(torch.nn.functional.cosine_similarity(zm, zb).detach())
        print(f"\n[{label}]  loss = {loss_fn.__doc__.splitlines()[0].strip()}")
        for t, c in traj:
            bar = "#" * int((c + 1) / 2 * 40)
            print(f"   step {t:4d}  cos(z_m,z_b) = {c:+.3f}  |{bar:<40}|")
        print(f"   -> converged cos = {final_cos:+.3f}")
        return final_cos

    print("=" * 70)
    print("E3 self-test — does each loss push complementary reps the right way?")
    print("=" * 70)
    c_buggy = optimize(l_dis_buggy, "BUGGY  (1 - cos)")
    c_fixed = optimize(l_dis_fixed, "FIXED  (1 + cos)/2")
    print("\n" + "-" * 70)
    print(f"buggy converged cos = {c_buggy:+.3f}  (expected -> +1, COLLAPSE/redundant)")
    print(f"fixed converged cos = {c_fixed:+.3f}  (expected -> -1, DISENTANGLED)")
    # Direction (sign) is the claim; cosine gradient vanishes near +/-1 so exact
    # saturation is step-dependent. Pass on clear, well-separated directions.
    ok = c_buggy > 0.5 and c_fixed < -0.5 and (c_buggy - c_fixed) > 1.2
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'} — fixed loss disentangles (cos<0), "
          f"buggy collapses (cos>0); separation={c_buggy - c_fixed:.2f}.")
    return ok


def main():
    ap = argparse.ArgumentParser(description="E3 loss-direction causal demo")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()
    _run_demo(steps=args.steps, dim=args.dim, lr=args.lr)


if __name__ == "__main__":
    main()
