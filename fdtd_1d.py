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
MU0  = 4.0e-7 * np.pi # permeability of free space (H/m)
EPS0 = 8.854187817e-12 # permittivity of free space (F/m)
# c is DERIVED from mu0 and eps0 rather than hard-coded to 3e8. The Mur ABC
# coefficient and the Courant number both compare c*dt against dx, so a c that
# disagrees with the mu0/eps0 used in the update coefficients by 0.07% detunes
# the boundary and leaves a spurious reflection behind.
C0   = 1.0 / np.sqrt(MU0 * EPS0)   # speed of light in vacuum (m/s)
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

    # Field probes — cells whose Ez time history is recorded every step.
    # These drive the reflection/transmission measurement in scattering.py.
    # Defaults (None) place them midway between source and slab, and midway
    # between the slab and the right boundary.
    probe_refl:   Optional[int] = None   # probe between source and slab
    probe_trans:  Optional[int] = None   # probe behind the slab

    # Derived quantities (computed post-init)
    dt:           float = field(init=False)

    def __post_init__(self):
        self.dt = self.courant * self.dx / C0
        if self.probe_refl is None:
            self.probe_refl = (self.source_pos + self.slab_start) // 2
        if self.probe_trans is None:
            self.probe_trans = (self.slab_end + self.num_cells - 1) // 2
        self.validate()

    def validate(self) -> None:
        """
        Check the layout before anything runs.

        Numpy slicing silently clips out-of-range indices, so a slab placed
        outside the grid used to produce a simulation with no slab in it and
        a plausible-looking report. Every position is checked explicitly, and
        the ordering source < probe < slab < probe is enforced because the
        two-run measurement depends on it: the reflection probe has to sit
        where only the reflected wave reaches it, and the transmission probe
        behind the slab.
        """
        if self.courant > 1.0:
            raise ValueError(
                f"courant={self.courant} violates the stability condition "
                f"(S <= 1); the simulation would diverge."
            )
        if not 0 < self.slab_start < self.slab_end <= self.num_cells:
            raise ValueError(
                f"slab spans cells {self.slab_start}..{self.slab_end}, which does "
                f"not fit a {self.num_cells}-cell grid. Pass --cells larger than "
                f"{self.slab_end}, or move the slab."
            )
        order = [
            ("source_pos", self.source_pos),
            ("probe_refl", self.probe_refl),
            ("slab_start", self.slab_start),
            ("slab_end", self.slab_end),
            ("probe_trans", self.probe_trans),
        ]
        for name, idx in order:
            if not 0 <= idx < self.num_cells:
                raise ValueError(
                    f"{name}={idx} is outside a {self.num_cells}-cell grid "
                    f"(valid 0..{self.num_cells - 1})"
                )
        for (lo_name, lo), (hi_name, hi) in zip(order, order[1:]):
            if lo >= hi:
                raise ValueError(
                    f"{lo_name}={lo} must be strictly left of {hi_name}={hi}. "
                    f"Required layout: source < probe_refl < slab_start < "
                    f"slab_end < probe_trans."
                )


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

        # Hy half a step behind the current one, kept so that magnetic energy
        # can be evaluated at the same instant as electric energy. Ez lives at
        # integer steps and Hy at half-integer steps, so summing them directly
        # produces a sawtooth that looks like an energy-conservation failure.
        self.Hy_prev = np.zeros(N)

        # ── Field probes (Ez time history at fixed cells) ────────────────────
        self.probe_cells = (cfg.probe_refl, cfg.probe_trans)
        self.probe_data = {c: np.zeros(cfg.num_steps) for c in self.probe_cells}


    def gaussian_source(self, step: int) -> float:
        """Gaussian pulse: E(t) = exp(-0.5 * ((t - delay) / width)^2)"""
        return np.exp(-0.5 * ((step - self.cfg.pulse_delay) / self.cfg.pulse_width) ** 2)
    
    def update_H(self):
        """Advance magnetic field by half a time step."""
        self.Hy_prev[:] = self.Hy
        self.Hy[:-1] += self.Db * (self.Ez[1:] - self.Ez[:-1])
    
    def update_E(self):
        """
        Advance electric field by one time step — INTERIOR CELLS ONLY.

        Ez[0] and Ez[-1] are boundary nodes owned by the Mur ABC, which needs
        them still holding their time-n values when it runs. Updating the last
        cell here (Ez[1:] rather than Ez[1:-1]) leaves the right-hand ABC
        differencing a time-(n+1) value against a time-n one, which detunes it
        into a near-perfect mirror: an empty grid then returned ~90% of the
        pulse amplitude off the right wall. The left boundary was accidentally
        correct because Ez[0] was already excluded.
        """
        self.Ez[1:-1] = self.Ca[1:-1] * self.Ez[1:-1] + \
                        self.Cb[1:-1] * (self.Hy[1:-1] - self.Hy[:-2])
    
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
        """
        Total EM energy in the grid (electric + magnetic).

        The magnetic term uses the average of Hy at n-1/2 and n+1/2 so that
        both terms are evaluated at time step n. Without this the leap-frog
        staggering shows up as a half-step ripple on the energy curve.
        """
        E_energy = 0.5 * np.sum(self.eps_profile * self.Ez ** 2) * self.cfg.dx
        Hy_centered = 0.5 * (self.Hy + self.Hy_prev)
        H_energy = 0.5 * MU0 * np.sum(Hy_centered ** 2) * self.cfg.dx
        return E_energy + H_energy

    def record_probes(self, n: int):
        """Store Ez at each probe cell for this time step."""
        for cell in self.probe_cells:
            self.probe_data[cell][n] = self.Ez[cell]

    def step(self, n: int):
        """Execute one complete FDTD time step."""
        self.update_H()
        self.update_E()
        self.apply_source(n)
        self.apply_mur_abc()
        self.record_probes(n)
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
# NOTE: an earlier version of this file measured R and T as
#   peak |Ez| left of the slab  /  peak |Ez| right of the slab
# taken from the FINAL field snapshot. That number is not a reflection
# coefficient. After the absorbing boundaries have drained the grid it decays
# towards zero, so the "measurement" was really a function of --steps, and it
# was reported next to a single-interface Fresnel coefficient it could not be
# compared against (field vs. peak amplitude, one interface vs. two).
#
# The measurement now lives in scattering.measure_scattering(), which runs the
# grid twice (empty reference + slab), subtracts to isolate the scattered
# field, and reports R(f)/T(f) across the pulse bandwidth.


def plot_validation(result: dict, cfg: SimConfig, output_dir: str = "results"):
    """
    Plot measured R(f) and T(f) against the closed-form slab solution.

    This is the plot that actually demonstrates the solver is correct: the
    Fabry-Perot ripple (the slab resonating at multiples of a half-wavelength)
    has to line up with theory, not just the average level.
    """
    os.makedirs(output_dir, exist_ok=True)

    f_ghz = result["freqs"] / 1e9
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=120, sharex=True)
    fig.suptitle(
        f"FDTD vs Analytic Slab Solution  (eps_r={cfg.eps_r}, "
        f"d={(cfg.slab_end - cfg.slab_start) * cfg.dx * 1e3:.0f} mm, "
        f"sigma={cfg.sigma} S/m)",
        fontsize=13, fontweight="bold")

    ax1.plot(f_ghz, result["R_analytic_f"], color="#111827", linewidth=2.5,
             alpha=0.45, label="Analytic |R(f)|  (Airy / Fabry-Perot)")
    ax1.plot(f_ghz, result["R_f"], color="#EF4444", linewidth=1.3,
             label="FDTD |R(f)|  (two-run measurement)")
    ax1.set_ylabel("|R|")
    ax1.set_title(f"Reflection — max error {result['max_R_error']:.4f}", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(f_ghz, result["T_analytic_f"], color="#111827", linewidth=2.5,
             alpha=0.45, label="Analytic |T(f)|")
    ax2.plot(f_ghz, result["T_f"], color="#2563EB", linewidth=1.3,
             label="FDTD |T(f)|")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("|T|")
    ax2.set_title(f"Transmission — max error {result['max_T_error']:.4f}", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "validation_spectra.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path


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
    report.append("3. REFLECTION / TRANSMISSION (two-run measurement)")
    lo, hi = rt_data["band_hz"]
    report.append(f"   Measurement band  : {lo/1e9:.2f} – {hi/1e9:.2f} GHz")
    report.append(f"   Recording length  : {rt_data['num_steps']} steps")
    report.append(f"   Reflected power   : {rt_data['R_power']*100:.2f} %")
    report.append(f"   Transmitted power : {rt_data['T_power']*100:.2f} %")
    report.append(f"   R + T             : {rt_data['power_sum']*100:.2f} %"
                  f"   (must be 100 % for a lossless slab)")
    report.append("")
    report.append("4. VALIDATION AGAINST CLOSED-FORM SLAB SOLUTION")
    report.append(f"   Analytic R power  : {rt_data['R_power_analytic']*100:.2f} %")
    report.append(f"   Analytic T power  : {rt_data['T_power_analytic']*100:.2f} %")
    report.append(f"   max |R_fdtd - R_theory| : {rt_data['max_R_error']:.5f}")
    report.append(f"   max |T_fdtd - T_theory| : {rt_data['max_T_error']:.5f}")
    report.append(f"   rms |R_fdtd - R_theory| : {rt_data['rms_R_error']:.5f}")
    report.append(f"   rms |T_fdtd - T_theory| : {rt_data['rms_T_error']:.5f}")
    report.append("   Reference: Airy (Fabry-Perot) summation over the slab's")
    report.append("   infinite series of internal reflections.")
    report.append("")
    report.append("5. ENERGY BOOKKEEPING")
    peak_e = max(engine.energy_history)
    final_e = engine.energy_history[-1]
    report.append(f"   Peak energy       : {peak_e:.6e} J")
    report.append(f"   Energy still in grid after {cfg.num_steps} steps : "
                  f"{(final_e / peak_e * 100) if peak_e > 0 else 0:.2f} %")
    report.append(f"   ... after {rt_data['num_steps']} steps (measurement run) : "
                  f"{rt_data['residual_energy_fraction'] * 100:.4f} %")
    report.append("   Energy is not conserved inside the grid by design: the")
    report.append("   absorbing boundaries carry it out. The measurement run is")
    report.append("   long enough that what remains is negligible, which is what")
    report.append("   makes the spectra above trustworthy.")
    report.append("")
    report.append("6. NUMERICAL METHOD")
    report.append("   Algorithm         : Yee FDTD (Finite-Difference Time-Domain)")
    report.append("   Boundary cond.    : First-order Mur Absorbing BC")
    report.append("   Source type       : Gaussian pulse (soft / additive)")
    report.append(f"   Stability         : Courant S = {cfg.courant} <= 1 (satisfied)")
    report.append(f"   Dispersion limit  : band capped at >= 20 cells/wavelength")
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

    # Configure. A bad layout (e.g. --cells smaller than the slab position)
    # is a usage error, so report it as one rather than as a traceback.
    try:
        cfg = SimConfig(
            num_cells=args.cells,
            num_steps=args.steps,
            eps_r=args.eps_r,
            sigma=args.sigma,
        )
    except ValueError as exc:
        parser.error(str(exc))

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

    # Reflection/Transmission measurement — needs its own, longer pair of runs
    # (empty reference + slab) so the scattered field can be isolated and the
    # slab's internal reverberation has time to leak out.
    from scattering import measure_scattering, default_num_steps
    meas_steps = max(default_num_steps(cfg), cfg.num_steps)
    print(f"  Measuring R/T (2 runs x {meas_steps} steps) ...")
    rt_data = measure_scattering(cfg, num_steps=meas_steps)
    plot_validation(rt_data, cfg, args.output)
    print()

    # Report
    print("  Generating technical report ...")
    generate_report(engine, rt_data, args.output)
    
    print()
    print(f"  All outputs saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
