"""
parameter_sweep.py
------------------
Automated engineering design study: sweeps the dielectric constant (eps_r) of
the slab and records how much power the slab reflects and transmits.

Every number plotted here is *measured* from the FDTD grid by the two-run
method in scattering.py. The closed-form slab solution is overlaid as a
validation curve, not used as the data.

This demonstrates:
  - Engineering workflow automation using Python
  - Parametric simulation studies
  - Validating a numerical solver against theory across a parameter range
  - The kind of design exploration EDA tools perform
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from fdtd_1d import SimConfig
from scattering import measure_scattering


def sweep_permittivity(eps_r_values: np.ndarray) -> list:
    """Run the two-run scattering measurement for each slab permittivity."""
    results = []

    print(f"  {'eps_r':>6} {'R meas':>9} {'R theory':>9} "
          f"{'T meas':>9} {'T theory':>9} {'R+T':>9}")
    print("  " + "-" * 60)

    for eps_r in eps_r_values:
        cfg = SimConfig(eps_r=float(eps_r))
        res = measure_scattering(cfg)
        res["eps_r"] = float(eps_r)
        results.append(res)

        print(f"  {eps_r:6.1f} {res['R_power']:9.4f} {res['R_power_analytic']:9.4f} "
              f"{res['T_power']:9.4f} {res['T_power_analytic']:9.4f} "
              f"{res['power_sum']:9.5f}")

    return results


def plot_sweep(results: list, output_dir: str = "results"):
    """Visualise the parameter sweep: measurement vs theory, and the residual."""
    os.makedirs(output_dir, exist_ok=True)

    eps = np.array([r["eps_r"] for r in results])
    R = np.array([r["R_power"] for r in results])
    T = np.array([r["T_power"] for r in results])
    Ra = np.array([r["R_power_analytic"] for r in results])
    Ta = np.array([r["T_power_analytic"] for r in results])
    total = R + T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle("Parametric Sweep: Dielectric Constant vs Band-Averaged Power Split",
                 fontsize=13, fontweight="bold")

    # ── Left: measured vs analytic ───────────────────────────────────────────
    ax1.plot(eps, Ra, "-", color="#111827", alpha=0.4, linewidth=3,
             label="Reflected (theory)")
    ax1.plot(eps, Ta, "--", color="#111827", alpha=0.4, linewidth=3,
             label="Transmitted (theory)")
    ax1.plot(eps, R, "o", color="#EF4444", markersize=7, label="Reflected (FDTD)")
    ax1.plot(eps, T, "s", color="#10B981", markersize=7, label="Transmitted (FDTD)")
    ax1.set_xlabel("Relative Permittivity (eps_r)")
    ax1.set_ylabel("Fraction of incident power")
    ax1.set_title("Measured power split vs closed-form solution")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Right: the two error checks that matter ──────────────────────────────
    ax2.axhline(0.0, color="#9CA3AF", linewidth=1)
    ax2.plot(eps, R - Ra, "o-", color="#EF4444", markersize=6,
             label="R error (FDTD - theory)")
    ax2.plot(eps, T - Ta, "s-", color="#10B981", markersize=6,
             label="T error (FDTD - theory)")
    ax2.plot(eps, total - 1.0, "^-", color="#2563EB", markersize=6,
             label="(R + T) - 1  [energy conservation]")
    ax2.set_xlabel("Relative Permittivity (eps_r)")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("Solver error against theory and against unity")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "parameter_sweep.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="FDTD parametric sweep over eps_r")
    ap.add_argument("--output", default="results", help="Output directory")
    args = ap.parse_args()

    print()
    print("=" * 60)
    print("  PARAMETRIC SWEEP: Dielectric Constant vs Wave Behaviour")
    print("=" * 60)
    print()

    eps_r_values = np.arange(1.5, 10.0, 1.0)
    print(f"  Sweeping eps_r = {[round(v, 1) for v in eps_r_values.tolist()]}")
    print(f"  Each point is 2 FDTD runs (empty reference + slab).\n")

    results = sweep_permittivity(eps_r_values)
    plot_sweep(results, args.output)

    worst_R = max(abs(r["R_power"] - r["R_power_analytic"]) for r in results)
    worst_sum = max(abs(r["power_sum"] - 1.0) for r in results)

    print()
    print("=" * 60)
    print("  SWEEP COMPLETE")
    print(f"  Worst |R_measured - R_theory| : {worst_R:.5f}")
    print(f"  Worst |(R + T) - 1|           : {worst_sum:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
