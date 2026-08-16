"""
parameter_sweep.py
------------------
Automated engineering design study: sweeps the dielectric constant (eps_r)
of the slab from 1.0 to 9.0 and records how reflection and transmission
change. Generates a comparison plot.

This demonstrates:
  - Engineering workflow automation using Python
  - Parametric simulation studies
  - Automated result aggregation and visualization
  - The kind of design exploration Keysight EDA tools perform
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from fdtd_1d import SimConfig, FDTDEngine, compute_reflection_transmission


def sweep_permittivity(eps_r_values: np.ndarray, num_steps: int = 2000):
    """Run multiple FDTD simulations with different slab permittivities."""
    results = []
    
    for eps_r in eps_r_values:
        cfg = SimConfig(eps_r=float(eps_r), num_steps=num_steps)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=num_steps)  # no snapshots needed
        
        rt = compute_reflection_transmission(engine)
        rt["eps_r"] = eps_r
        results.append(rt)
        
        print(f"  eps_r = {eps_r:.1f}  |  R_analytical = {rt['R_analytical']:.4f}  "
              f"|  Transmitted = {rt['peak_transmitted_numerical']:.4f}")
    
    return results


def plot_sweep(results: list, output_dir: str = "results"):
    """Visualize the parameter sweep results."""
    os.makedirs(output_dir, exist_ok=True)
    
    eps_vals = [r["eps_r"] for r in results]
    R_analytical = [abs(r["R_analytical"]) for r in results]
    T_analytical = [abs(r["T_analytical"]) for r in results]
    peak_trans = [r["peak_transmitted_numerical"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle("Parametric Sweep: Effect of Dielectric Constant on EM Wave Interaction",
                 fontsize=13, fontweight="bold")
    
    # Analytical Fresnel coefficients
    ax1.plot(eps_vals, R_analytical, "o-", color="#EF4444", linewidth=2,
             markersize=6, label="|R| (Fresnel)")
    ax1.plot(eps_vals, T_analytical, "s-", color="#10B981", linewidth=2,
             markersize=6, label="|T| (Fresnel)")
    ax1.set_xlabel("Relative Permittivity (eps_r)")
    ax1.set_ylabel("Coefficient Magnitude")
    ax1.set_title("Analytical Fresnel Coefficients")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 9)
    
    # Numerical transmitted peak
    ax2.bar(eps_vals, peak_trans, width=0.4, color="#2563EB", alpha=0.8,
            edgecolor="#1E40AF")
    ax2.set_xlabel("Relative Permittivity (eps_r)")
    ax2.set_ylabel("Peak Transmitted Ez (V/m)")
    ax2.set_title("FDTD Numerical Transmitted Amplitude")
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "parameter_sweep.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {path}")
    return path


def main():
    print()
    print("=" * 60)
    print("  PARAMETRIC SWEEP: Dielectric Constant vs Wave Behaviour")
    print("=" * 60)
    print()
    
    eps_r_values = np.arange(1.0, 10.0, 1.0)
    print(f"  Sweeping eps_r = {eps_r_values.tolist()}")
    print(f"  Running {len(eps_r_values)} simulations ...\n")
    
    results = sweep_permittivity(eps_r_values)
    plot_sweep(results)
    
    print("\n" + "=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
