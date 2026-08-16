"""
fdtd_1d.py
----------
1-D Finite-Difference Time-Domain (FDTD) Electromagnetic Wave Simulator.

Simulates how an EM pulse propagates through free space and interacts with
a dielectric slab (e.g., glass, PCB substrate, biological tissue).  Uses the
exact same Yee-grid update equations that commercial tools like Keysight ADS,
Ansys HFSS, and CST Studio solve — just in one spatial dimension for clarity.

Physics implemented
-------------------
* Maxwell's curl equations discretised on a staggered Yee grid:
    dEz/dt =  (1/eps) * dHy/dx
    dHy/dt =  (1/mu)  * dEz/dx
* Gaussian pulse soft source (additive)
* First-order Mur absorbing boundary conditions (ABC)
* Dielectric slab with user-defined relative permittivity & loss tangent
* Energy conservation tracking (validates numerical stability)

Author : Nitesh  |  NSUT Delhi  |  2026
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import argparse
import os

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════════
C0   = 3.0e8          # speed of light in vacuum  (m/s)
MU0  = 4.0e-7 * np.pi # permeability of free space (H/m)
EPS0 = 8.854187817e-12 # permittivity of free space (F/m)
ETA0 = np.sqrt(MU0 / EPS0)  # impedance of free space (~377 ohms)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation domain configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimConfig:
    """All parameters that define a 1-D FDTD simulation."""
    # Grid
    num_cells:    int   = 500       # number of spatial cells
    dx:           float = 1e-3      # cell size (m) — 1 mm
    
    # Time stepping (Courant number ≤ 1 for stability)
    courant:      float = 0.5       # Courant number S = c*dt/dx
    num_steps:    int   = 1500      # total time steps
    
    # Source — Gaussian pulse
    source_pos:   int   = 100       # cell index of the source
    pulse_width:  float = 30.0      # pulse width (in time steps)
    pulse_delay:  float = 80.0      # pulse peak delay (in time steps)
    
    # Dielectric slab
    slab_start:   int   = 250       # cell where slab begins
    slab_end:     int   = 350       # cell where slab ends
    eps_r:        float = 4.0       # relative permittivity (e.g., FR-4 PCB substrate ≈ 4.0)
    sigma:        float = 0.0       # conductivity (S/m), 0 = lossless
    
    # Derived quantities (computed post-init)
    dt:           float = field(init=False)
    
    def __post_init__(self):
        self.dt = self.courant * self.dx / C0


# ═══════════════════════════════════════════════════════════════════════════════
# FDTD Engine
# ═══════════════════════════════════════════════════════════════════════════════
class FDTDEngine:
    """
    Core 1-D FDTD solver.
    
    Implements the Yee algorithm:
      1. Update H-field (half time step ahead of E)
      2. Update E-field
      3. Apply source
      4. Apply absorbing boundary conditions
    """
    
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        N = cfg.num_cells
        
        # ── Field arrays (staggered Yee grid) ────────────────────────────────
        self.Ez = np.zeros(N)    # Electric field (z-component)
        self.Hy = np.zeros(N)    # Magnetic field (y-component)
        
        # ── Material arrays ──────────────────────────────────────────────────
        self.eps_profile = np.ones(N) * EPS0           # permittivity at each cell
        self.sigma_profile = np.zeros(N)               # conductivity at each cell
        
        # Fill the dielectric slab region
        self.eps_profile[cfg.slab_start:cfg.slab_end] = cfg.eps_r * EPS0
        self.sigma_profile[cfg.slab_start:cfg.slab_end] = cfg.sigma
        
        # ── Update coefficients (pre-computed for speed) ─────────────────────
        # These come from discretising Maxwell's equations:
        #   Ez^{n+1} = Ca * Ez^n + Cb * (Hy^{n+1/2}[i] - Hy^{n+1/2}[i-1])
        self.Ca = (1 - cfg.dt * self.sigma_profile / (2 * self.eps_profile)) / \
                  (1 + cfg.dt * self.sigma_profile / (2 * self.eps_profile))
        self.Cb = (cfg.dt / (self.eps_profile * cfg.dx)) / \
                  (1 + cfg.dt * self.sigma_profile / (2 * self.eps_profile))
        
        # For H update (non-magnetic medium, mu = mu0 everywhere):
        #   Hy^{n+1/2} = Hy^{n-1/2} + (dt / (mu0 * dx)) * (Ez^n[i+1] - Ez^n[i])
        self.Db = cfg.dt / (MU0 * cfg.dx)
        
        # ── Mur ABC storage (previous boundary values) ───────────────────────
        self.Ez_left_prev  = 0.0
        self.Ez_right_prev = 0.0
        
        # ── Diagnostics ──────────────────────────────────────────────────────
        self.energy_history = []
        self.snapshots = []           # (step, Ez_copy) tuples
    
    def gaussian_source(self, step: int) -> float:
        """Gaussian pulse: E(t) = exp(-0.5 * ((t - delay) / width)^2)"""
        return np.exp(-0.5 * ((step - self.cfg.pulse_delay) / self.cfg.pulse_width) ** 2)
    
    def update_H(self):
        """Advance magnetic field by half a time step."""
        self.Hy[:-1] += self.Db * (self.Ez[1:] - self.Ez[:-1])
    
    def update_E(self):
        """Advance electric field by one time step."""
        self.Ez[1:] = self.Ca[1:] * self.Ez[1:] + \
                      self.Cb[1:] * (self.Hy[1:] - self.Hy[:-1])
    
    def apply_source(self, step: int):
        """Inject Gaussian pulse as a soft source (additive)."""
        self.Ez[self.cfg.source_pos] += self.gaussian_source(step)
    
    def apply_mur_abc(self):
        """First-order Mur absorbing boundary conditions.
        
        Prevents artificial reflections from the grid edges by
        approximating an outgoing wave condition.
        """
        coeff = (C0 * self.cfg.dt - self.cfg.dx) / (C0 * self.cfg.dt + self.cfg.dx)
        
        # Left boundary
        self.Ez[0] = self.Ez_left_prev + coeff * (self.Ez[1] - self.Ez[0])
        self.Ez_left_prev = self.Ez[1]
        
        # Right boundary
        self.Ez[-1] = self.Ez_right_prev + coeff * (self.Ez[-2] - self.Ez[-1])
        self.Ez_right_prev = self.Ez[-2]
    
    def compute_energy(self) -> float:
        """Total EM energy in the grid (electric + magnetic)."""
        E_energy = 0.5 * np.sum(self.eps_profile * self.Ez ** 2) * self.cfg.dx
        H_energy = 0.5 * MU0 * np.sum(self.Hy ** 2) * self.cfg.dx
        return E_energy + H_energy
    
    def step(self, n: int):
        """Execute one complete FDTD time step."""
        self.update_H()
        self.update_E()
        self.apply_source(n)
        self.apply_mur_abc()
        self.energy_history.append(self.compute_energy())
    
    def run(self, snapshot_interval: int = 50):
        """Run the full simulation."""
        for n in range(self.cfg.num_steps):
            self.step(n)
            if n % snapshot_interval == 0:
                self.snapshots.append((n, self.Ez.copy()))
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def plot_snapshots(engine: FDTDEngine, output_dir: str = "results"):
    """Generate a multi-panel plot showing wave propagation at key time steps."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = engine.cfg
    x_mm = np.arange(cfg.num_cells) * cfg.dx * 1000  # convert to mm
    
    # Pick 6 evenly-spaced snapshots
    indices = np.linspace(0, len(engine.snapshots) - 1, 6, dtype=int)
    chosen = [engine.snapshots[i] for i in indices]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=120)
    fig.suptitle("1-D FDTD: EM Pulse Propagation Through Dielectric Slab",
                 fontsize=14, fontweight="bold")
    
    for ax, (step, Ez) in zip(axes.flat, chosen):
        ax.plot(x_mm, Ez, color="#2563EB", linewidth=1.2, label="Ez field")
        
        # Shade the dielectric slab
        ax.axvspan(cfg.slab_start * cfg.dx * 1000,
                   cfg.slab_end * cfg.dx * 1000,
                   alpha=0.15, color="#F59E0B", label=f"Slab (eps_r={cfg.eps_r})")
        
        ax.set_title(f"Step {step}  (t = {step * cfg.dt * 1e12:.1f} ps)", fontsize=10)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Ez (V/m)")
        ax.set_ylim(-1.2, 1.2)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "wave_propagation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path


def plot_energy(engine: FDTDEngine, output_dir: str = "results"):
    """Plot total EM energy over time (validates numerical stability)."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    steps = np.arange(len(engine.energy_history))
    ax.plot(steps, engine.energy_history, color="#10B981", linewidth=1.5)
    ax.set_title("Total Electromagnetic Energy vs Time Step", fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Energy (J)")
    ax.grid(True, alpha=0.3)
    
    # Check energy conservation (should plateau after pulse enters, then decay
    # only if there's loss or energy leaves through ABCs)
    peak = max(engine.energy_history)
    final = engine.energy_history[-1]
    ax.annotate(f"Peak: {peak:.2e} J\nFinal: {final:.2e} J",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#10B981"))
    
    plt.tight_layout()
    path = os.path.join(output_dir, "energy_conservation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path


def plot_material_profile(cfg: SimConfig, output_dir: str = "results"):
    """Plot the spatial permittivity profile of the simulation domain."""
    os.makedirs(output_dir, exist_ok=True)
    
    x_mm = np.arange(cfg.num_cells) * cfg.dx * 1000
    eps_r_profile = np.ones(cfg.num_cells)
    eps_r_profile[cfg.slab_start:cfg.slab_end] = cfg.eps_r
    
    fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
    ax.fill_between(x_mm, eps_r_profile, alpha=0.4, color="#8B5CF6")
    ax.plot(x_mm, eps_r_profile, color="#8B5CF6", linewidth=1.5)
    ax.set_title("Relative Permittivity Profile (Simulation Domain)", fontweight="bold")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("eps_r")
    ax.set_ylim(0, cfg.eps_r + 1)
    ax.grid(True, alpha=0.3)
    
    # Annotate regions
    ax.annotate("Free Space\n(eps_r = 1.0)", xy=(50, 0.5), fontsize=9, color="#6B7280")
    ax.annotate(f"Dielectric Slab\n(eps_r = {cfg.eps_r})",
                xy=(cfg.slab_start * cfg.dx * 1000 + 10, cfg.eps_r - 0.5),
                fontsize=9, color="#5B21B6", fontweight="bold")
    ax.annotate("Free Space\n(eps_r = 1.0)",
                xy=(cfg.slab_end * cfg.dx * 1000 + 20, 0.5), fontsize=9, color="#6B7280")
    
    # Mark source position
    ax.axvline(cfg.source_pos * cfg.dx * 1000, color="#EF4444", linestyle="--",
               linewidth=1, label="Pulse Source")
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "material_profile.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Reflection & Transmission Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def compute_reflection_transmission(engine: FDTDEngine) -> dict:
    """
    Estimate reflection and transmission coefficients.
    
    Theory (for a lossless slab at normal incidence):
      R = (n1 - n2) / (n1 + n2)   at each interface
      where n = sqrt(eps_r) is the refractive index.
    
    We compare the analytical Fresnel coefficient with the
    numerically observed reflected/transmitted pulse amplitudes.
    """
    cfg = engine.cfg
    n1, n2 = 1.0, np.sqrt(cfg.eps_r)
    
    # Analytical single-interface Fresnel reflection
    R_analytical = (n1 - n2) / (n1 + n2)
    T_analytical = 2 * n1 / (n1 + n2)
    
    # Numerical: measure peak Ez in reflected and transmitted regions
    # (after the simulation has run long enough for the pulse to fully interact)
    reflected_region = engine.Ez[:cfg.slab_start]
    transmitted_region = engine.Ez[cfg.slab_end:]
    
    peak_reflected = np.max(np.abs(reflected_region)) if len(reflected_region) > 0 else 0
    peak_transmitted = np.max(np.abs(transmitted_region)) if len(transmitted_region) > 0 else 0
    
    return {
        "n1": n1,
        "n2": n2,
        "R_analytical": R_analytical,
        "T_analytical": T_analytical,
        "peak_reflected_numerical": peak_reflected,
        "peak_transmitted_numerical": peak_transmitted,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════════
def generate_report(engine: FDTDEngine, rt_data: dict, output_dir: str = "results"):
    """Write a technical summary report."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = engine.cfg
    
    report = []
    report.append("=" * 65)
    report.append("  1-D FDTD ELECTROMAGNETIC WAVE SIMULATION — TECHNICAL REPORT")
    report.append("=" * 65)
    report.append("")
    report.append("1. SIMULATION PARAMETERS")
    report.append(f"   Grid cells       : {cfg.num_cells}")
    report.append(f"   Cell size (dx)    : {cfg.dx * 1e3:.2f} mm")
    report.append(f"   Time steps        : {cfg.num_steps}")
    report.append(f"   Time step (dt)    : {cfg.dt * 1e12:.4f} ps")
    report.append(f"   Courant number    : {cfg.courant}")
    report.append(f"   Source position   : cell {cfg.source_pos}")
    report.append(f"   Pulse width       : {cfg.pulse_width} steps")
    report.append("")
    report.append("2. MATERIAL CONFIGURATION")
    report.append(f"   Slab region       : cells {cfg.slab_start} – {cfg.slab_end}")
    report.append(f"   Slab thickness    : {(cfg.slab_end - cfg.slab_start) * cfg.dx * 1e3:.1f} mm")
    report.append(f"   Relative eps (er) : {cfg.eps_r}")
    report.append(f"   Conductivity      : {cfg.sigma} S/m")
    report.append(f"   Refractive index  : n = {np.sqrt(cfg.eps_r):.3f}")
    report.append("")
    report.append("3. REFLECTION / TRANSMISSION ANALYSIS")
    report.append(f"   Analytical Fresnel R   : {rt_data['R_analytical']:.4f}")
    report.append(f"   Analytical Fresnel T   : {rt_data['T_analytical']:.4f}")
    report.append(f"   Numerical peak (refl)  : {rt_data['peak_reflected_numerical']:.6f}")
    report.append(f"   Numerical peak (trans) : {rt_data['peak_transmitted_numerical']:.6f}")
    report.append("")
    report.append("4. ENERGY CONSERVATION")
    peak_e = max(engine.energy_history)
    final_e = engine.energy_history[-1]
    report.append(f"   Peak energy       : {peak_e:.6e} J")
    report.append(f"   Final energy      : {final_e:.6e} J")
    report.append(f"   Energy retention  : {(final_e / peak_e * 100) if peak_e > 0 else 0:.1f}%")
    report.append(f"   (Energy leaves through ABCs — expected behaviour)")
    report.append("")
    report.append("5. NUMERICAL METHOD")
    report.append("   Algorithm         : Yee FDTD (Finite-Difference Time-Domain)")
    report.append("   Boundary cond.    : First-order Mur Absorbing BC")
    report.append("   Source type       : Gaussian pulse (soft / additive)")
    report.append("   Stability         : Courant condition satisfied (S <= 1)")
    report.append("")
    report.append("=" * 65)
    
    text = "\n".join(report)
    path = os.path.join(output_dir, "simulation_report.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  [SAVED] {path}")
    
    # Also print to terminal
    print()
    print(text)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="1-D FDTD Electromagnetic Wave Simulator"
    )
    parser.add_argument("--cells", type=int, default=500,
                        help="Number of grid cells (default: 500)")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Number of time steps (default: 1500)")
    parser.add_argument("--eps-r", type=float, default=4.0,
                        help="Relative permittivity of the slab (default: 4.0 for FR-4)")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Conductivity of the slab in S/m (default: 0.0 lossless)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots and report")
    args = parser.parse_args()
    
    # Configure
    cfg = SimConfig(
        num_cells=args.cells,
        num_steps=args.steps,
        eps_r=args.eps_r,
        sigma=args.sigma,
    )
    
    print()
    print("=" * 60)
    print("  1-D FDTD ELECTROMAGNETIC WAVE SIMULATOR")
    print("=" * 60)
    print(f"  Grid: {cfg.num_cells} cells x {cfg.dx*1e3:.1f} mm = {cfg.num_cells * cfg.dx * 1e3:.0f} mm domain")
    print(f"  Time: {cfg.num_steps} steps x {cfg.dt*1e12:.2f} ps = {cfg.num_steps * cfg.dt * 1e9:.2f} ns")
    print(f"  Slab: eps_r={cfg.eps_r}, sigma={cfg.sigma} S/m")
    print(f"  Courant number: {cfg.courant} (stable)")
    print("=" * 60)
    print()
    
    # Run simulation
    print("  Running FDTD simulation ...")
    engine = FDTDEngine(cfg)
    engine.run(snapshot_interval=50)
    print(f"  Done. {len(engine.snapshots)} snapshots captured.")
    print()
    
    # Generate outputs
    print("  Generating visualizations ...")
    plot_material_profile(cfg, args.output)
    plot_snapshots(engine, args.output)
    plot_energy(engine, args.output)
    print()
    
    # Reflection/Transmission analysis
    rt_data = compute_reflection_transmission(engine)
    
    # Report
    print("  Generating technical report ...")
    generate_report(engine, rt_data, args.output)
    
    print()
    print(f"  All outputs saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
