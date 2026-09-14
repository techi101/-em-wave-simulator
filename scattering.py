"""
scattering.py
-------------
Broadband reflection / transmission measurement for the 1-D FDTD solver,
plus the closed-form slab solution used to validate it.

Why this module exists
----------------------
Reflection and transmission cannot be read off the final field snapshot.
By the time a simulation ends the pulse has largely left the grid through
the absorbing boundaries, so "peak |Ez| still sitting to the left of the
slab" is a function of how long you ran, not of the physics.

The standard fix is a *two-run* (total-field / scattered-field) measurement:

    run 1 (reference) : empty grid, no slab   -> gives the INCIDENT field
    run 2 (total)     : same grid, with slab  -> gives the TOTAL field

    scattered = total - incident

A probe between the source and the slab records the reflected wave (after
subtracting the incident), and a probe behind the slab records the
transmitted wave. Both are normalised by the incident spectrum, which turns
a single pair of runs into a broadband frequency sweep:

    R(f) = |E_refl(f)|  / |E_inc(f)|
    T(f) = |E_trans(f)| / |E_inc(f)|

Because the Gaussian pulse is wideband, one pair of runs yields R and T
across several GHz - and the Fabry-Perot ripple of the slab is resolved,
which is a far stronger correctness check than a single-interface Fresnel
number.

Sign convention: the analytic formulas below use exp(-i*omega*t), so a lossy
medium has eps_c = eps_r + i*sigma/(omega*eps0) and Im(n) >= 0 (decay).
Only magnitudes are compared against FDTD, so the convention cannot leak
into the validation result.
"""

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from fdtd_1d import C0, EPS0, SimConfig, FDTDEngine


# Frequencies with fewer than this many cells per wavelength inside the slab
# are discarded: FDTD numerical dispersion, not the physics, dominates there.
MIN_CELLS_PER_WAVELENGTH = 20.0

# Frequencies where the incident pulse carries less than this fraction of its
# peak spectral amplitude are discarded: R = |E_r|/|E_i| is numerically
# meaningless when the denominator is ~0.
MIN_SPECTRAL_FRACTION = 0.01


# =============================================================================
# Closed-form solution for a dielectric slab in free space (normal incidence)
# =============================================================================
def analytic_slab_rt(freqs: np.ndarray,
                     eps_r: float,
                     thickness: float,
                     sigma: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Field reflection and transmission coefficients of a slab of thickness
    `thickness` (m) and relative permittivity `eps_r` embedded in free space.

    This is the Airy / Fabry-Perot summation of the infinite series of
    internal reflections:

        r = r12 (1 - P^2) / (1 - r12^2 P^2)
        t = t12 t23 P     / (1 - r12^2 P^2)

    with r12 = (1 - n)/(1 + n) and P = exp(i k0 n d) the one-way phase
    across the slab.

    `t` is returned *relative to free-space propagation over the same
    distance*, so it can be compared directly with an FDTD measurement that
    normalises by an empty-grid reference run.

    Returns:
        (r, t) complex arrays, same shape as `freqs`.
    """
    freqs = np.asarray(freqs, dtype=float)
    omega = 2.0 * np.pi * freqs

    # Complex relative permittivity (exp(-i omega t) convention).
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_c = eps_r + 1j * sigma / (omega * EPS0)
    # DC limit: guard the division instead of emitting a NaN into the
    # comparison. f = 0 is excluded from the measurement band anyway.
    eps_c = np.where(omega == 0, eps_r + 0j, eps_c)

    n = np.sqrt(eps_c)                      # principal branch -> Im(n) >= 0
    k0 = omega / C0

    P = np.exp(1j * k0 * n * thickness)     # one-way phase through the slab
    r12 = (1.0 - n) / (1.0 + n)
    t12 = 2.0 / (1.0 + n)
    t23 = 2.0 * n / (n + 1.0)

    denom = 1.0 - (r12 ** 2) * (P ** 2)
    r = r12 * (1.0 - P ** 2) / denom
    t = t12 * t23 * P / denom

    # Reference the transmission to free-space propagation over the slab,
    # matching what the two-run FDTD measurement reports.
    t = t * np.exp(-1j * k0 * thickness)
    return r, t


# =============================================================================
# Reference-run cache
# =============================================================================
_REFERENCE_CACHE: dict = {}


def _reference_key(cfg: SimConfig, num_steps: int) -> tuple:
    """Everything that changes the empty-grid incident field."""
    return (cfg.num_cells, cfg.dx, cfg.courant, num_steps,
            cfg.source_pos, cfg.pulse_width, cfg.pulse_delay,
            cfg.probe_refl, cfg.probe_trans)


def _run_probed(cfg: SimConfig, num_steps: int) -> FDTDEngine:
    """Run a simulation for `num_steps` with probe recording and no snapshots."""
    run_cfg = replace(cfg, num_steps=num_steps)
    engine = FDTDEngine(run_cfg)
    engine.run(snapshot_interval=num_steps + 1)   # suppress snapshot capture
    return engine


def _incident_run(cfg: SimConfig, num_steps: int) -> FDTDEngine:
    """Empty-grid reference run (cached: it does not depend on the slab)."""
    key = _reference_key(cfg, num_steps)
    if key not in _REFERENCE_CACHE:
        empty = replace(cfg, eps_r=1.0, sigma=0.0)
        _REFERENCE_CACHE[key] = _run_probed(empty, num_steps)
    return _REFERENCE_CACHE[key]


def default_num_steps(cfg: SimConfig) -> int:
    """
    Pick a run length that lets the pulse fully clear the grid, including the
    multiply-reflected reverberation trapped inside the slab.

    Budget: two domain transits, plus ~12 round trips inside the slab (enough
    for the internal series to fall below the numerical noise floor), plus
    the Gaussian tail.
    """
    n_slab = np.sqrt(max(cfg.eps_r, 1.0))
    slab_cells = max(cfg.slab_end - cfg.slab_start, 1)
    transit = cfg.num_cells / cfg.courant
    reverb = 12.0 * 2.0 * slab_cells * n_slab / cfg.courant
    tail = 4.0 * cfg.pulse_width + cfg.pulse_delay
    return int(2 * transit + reverb + tail)


# =============================================================================
# The measurement
# =============================================================================
def measure_scattering(cfg: SimConfig,
                       num_steps: Optional[int] = None,
                       compare_analytic: bool = True) -> dict:
    """
    Measure broadband R(f) and T(f) for the slab described by `cfg`.

    Runs the simulation twice (empty reference + slab) and forms the
    scattered field by subtraction. See the module docstring.

    Returns a dict with:
        freqs        : frequency axis of the valid band (Hz)
        R_f, T_f     : measured field magnitudes |R(f)|, |T(f)| on that band
        R_power      : band-integrated reflected power fraction
        T_power      : band-integrated transmitted power fraction
        power_sum    : R_power + T_power (must be ~1 for a lossless slab)
        R_analytic_f : closed-form |r(f)| on the same band (if requested)
        T_analytic_f : closed-form |t(f)| on the same band (if requested)
        max_R_error  : max |R_fdtd - R_analytic| over the band
        max_T_error  : max |T_fdtd - T_analytic| over the band
        num_steps, band_hz, residual_energy_fraction
    """
    if num_steps is None:
        num_steps = default_num_steps(cfg)

    ref = _incident_run(cfg, num_steps)
    tot = _run_probed(cfg, num_steps)

    dt = cfg.dt
    inc_r = ref.probe_data[cfg.probe_refl]      # incident, at reflection probe
    inc_t = ref.probe_data[cfg.probe_trans]     # incident, at transmission probe
    tot_r = tot.probe_data[cfg.probe_refl]
    tot_t = tot.probe_data[cfg.probe_trans]

    reflected = tot_r - inc_r                   # scattered = total - incident
    transmitted = tot_t

    freqs = np.fft.rfftfreq(num_steps, d=dt)
    Ei_r = np.fft.rfft(inc_r)
    Ei_t = np.fft.rfft(inc_t)
    Er = np.fft.rfft(reflected)
    Et = np.fft.rfft(transmitted)

    # -- Restrict to the band where the measurement is meaningful ------------
    n_slab = np.sqrt(max(cfg.eps_r, 1.0))
    f_dispersion = C0 / (MIN_CELLS_PER_WAVELENGTH * cfg.dx * n_slab)
    amp = np.abs(Ei_r)
    band = ((amp >= MIN_SPECTRAL_FRACTION * amp.max())
            & (freqs <= f_dispersion)
            & (freqs > 0))

    if not np.any(band):
        raise RuntimeError(
            "No usable frequency band - check pulse width against grid size."
        )

    f_band = freqs[band]
    R_f = np.abs(Er[band]) / np.abs(Ei_r[band])
    T_f = np.abs(Et[band]) / np.abs(Ei_t[band])

    # -- Band-integrated power fractions (Parseval over the usable band) -----
    inc_power = np.sum(np.abs(Ei_r[band]) ** 2)
    R_power = float(np.sum(np.abs(Er[band]) ** 2) / inc_power)
    T_power = float(np.sum(np.abs(Et[band]) ** 2) / np.sum(np.abs(Ei_t[band]) ** 2))

    # How much energy was still bouncing around when we stopped recording?
    peak_energy = max(tot.energy_history) if tot.energy_history else 0.0
    residual = float(tot.energy_history[-1] / peak_energy) if peak_energy > 0 else 0.0

    out = {
        "freqs": f_band,
        "R_f": R_f,
        "T_f": T_f,
        "R_power": R_power,
        "T_power": T_power,
        "power_sum": R_power + T_power,
        "num_steps": num_steps,
        "band_hz": (float(f_band[0]), float(f_band[-1])),
        "residual_energy_fraction": residual,
        "eps_r": cfg.eps_r,
        "sigma": cfg.sigma,
    }

    if compare_analytic:
        thickness = (cfg.slab_end - cfg.slab_start) * cfg.dx
        r_a, t_a = analytic_slab_rt(f_band, cfg.eps_r, thickness, cfg.sigma)
        R_a, T_a = np.abs(r_a), np.abs(t_a)
        w = np.abs(Ei_r[band]) ** 2            # weight by incident spectral power
        out.update({
            "R_analytic_f": R_a,
            "T_analytic_f": T_a,
            "max_R_error": float(np.max(np.abs(R_f - R_a))),
            "max_T_error": float(np.max(np.abs(T_f - T_a))),
            "rms_R_error": float(np.sqrt(np.mean((R_f - R_a) ** 2))),
            "rms_T_error": float(np.sqrt(np.mean((T_f - T_a) ** 2))),
            "R_power_analytic": float(np.sum(R_a ** 2 * w) / np.sum(w)),
            "T_power_analytic": float(np.sum(T_a ** 2 * w) / np.sum(w)),
        })

    return out
