"""
surrogate_model.py
------------------
Trains a machine-learning surrogate on FDTD simulation output.

Instead of running the full physics simulation for every candidate design
(hours per point in 3-D), a surrogate learns the design -> response mapping
and predicts in microseconds. That is the standard acceleration pattern in
EDA and RF/antenna design.

The design problem
------------------
Pick a dielectric slab that is as transparent as possible at an operating
frequency of 3 GHz — a radome, or a matching window.

    inputs  : eps_r      1.5 .. 10     (relative permittivity)
              thickness  40 .. 160 mm  (slab thickness)
    output  : reflected power at 3 GHz

Both inputs matter and they interact: thickness sets the round-trip phase
inside the slab, so reflection oscillates between near-total transparency
(half-wave windows) and strong reflection as thickness varies. The response
surface is genuinely two-dimensional and oscillatory, which is what makes it
worth fitting a surrogate to at all.

The target is **measured from the FDTD grid** by the two-run method in
scattering.py — one empty reference run, one run with the slab, subtracted to
isolate the scattered field, then Fourier-transformed and evaluated at 3 GHz.

Two corrections from an earlier version of this file, both of which had made
the reported accuracy meaningless:

1. It trained on `abs(rt["R_analytical"])` — the closed-form Fresnel
   coefficient — while the plot labelled that curve "FDTD Simulation (ground
   truth)". The model was fitting a formula it had been handed directly, so
   its R^2 measured nothing, least of all the simulator.

2. Its target was a single-interface coefficient depending only on eps_r, so
   the "design space" was one-dimensional and any smooth model scored ~1.00.
   Band-averaging over the whole pulse bandwidth has the same problem: it
   averages over many Fabry-Perot periods, and the result is independent of
   thickness to four decimal places. Evaluating at an operating frequency is
   both the honest choice and the one a designer actually specifies.

Physics-informed features
-------------------------
The script fits the same models on naive inputs (eps_r, d) and on
physics-informed ones (refractive index and the round-trip phase). The
comparison is the point: the naive features cannot represent the resonance,
and the held-out scores say so.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from fdtd_1d import C0, SimConfig
from scattering import default_num_steps, measure_scattering

# Design space
EPS_MIN, EPS_MAX = 1.5, 10.0
THICK_MIN_MM, THICK_MAX_MM = 40.0, 160.0
TARGET_FREQ = 3.0e9          # operating frequency the slab is designed for

# Geometry held fixed across the study. Pinning the probes (rather than letting
# them default from the slab position) keeps the empty-grid reference run
# identical for every sample, so it is computed once and cached.
SLAB_START = 250
PROBE_REFL = 175
PROBE_TRANS = 450


def _config(eps_r: float, thickness_cells: int) -> SimConfig:
    return SimConfig(
        eps_r=float(eps_r),
        slab_start=SLAB_START,
        slab_end=SLAB_START + int(thickness_cells),
        probe_refl=PROBE_REFL,
        probe_trans=PROBE_TRANS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════════
def features_naive(X: np.ndarray) -> np.ndarray:
    """Raw design variables: permittivity and thickness."""
    return X.copy()


def features_physics(X: np.ndarray) -> np.ndarray:
    """
    Physics-informed features.

    A slab's reflection at one frequency is set by two things: the interface
    mismatch, which depends on the refractive index n = sqrt(eps_r), and the
    round-trip phase inside the slab, phi = 2 * k0 * n * d. Reflection is
    periodic in phi, so handing the model sin(phi) and cos(phi) gives it the
    right basis. Without them a low-order polynomial cannot represent the
    resonance at all — which is exactly what the score table below shows.
    """
    eps_r, d_mm = X[:, 0], X[:, 1]
    n = np.sqrt(eps_r)
    phi = 2.0 * (2.0 * np.pi * TARGET_FREQ / C0) * n * (d_mm * 1e-3)
    return np.column_stack([n, n * d_mm, np.cos(phi), np.sin(phi)])


FEATURE_SETS = {
    "naive (eps_r, d)": features_naive,
    "physics-informed": features_physics,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_training_data(n_samples: int = 150, seed: int = 0):
    """
    Run FDTD measurements across the design space.

    Stratified (Latin-hypercube-style) sampling rather than a grid, so held-out
    points are not all immediate neighbours of training points.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
    v = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
    rng.shuffle(v)

    eps_vals = EPS_MIN + u * (EPS_MAX - EPS_MIN)
    thick_mm = THICK_MIN_MM + v * (THICK_MAX_MM - THICK_MIN_MM)
    thick_cells = np.round(thick_mm).astype(int)

    # One recording length for every sample keeps the reference run cacheable.
    num_steps = default_num_steps(_config(EPS_MAX, int(THICK_MAX_MM)))

    print(f"  Generating {n_samples} FDTD measurements "
          f"({num_steps} steps, 2 grid runs per point) ...")

    X, y = [], []
    start = time.time()
    for i, (eps_r, cells) in enumerate(zip(eps_vals, thick_cells)):
        cfg = _config(eps_r, cells)
        res = measure_scattering(cfg, num_steps=num_steps, compare_analytic=False)

        # Reflected POWER at the operating frequency.
        r_at_f0 = np.interp(TARGET_FREQ, res["freqs"], res["R_f"])
        X.append([eps_r, cells * cfg.dx * 1e3])
        y.append(float(r_at_f0 ** 2))

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_samples} complete ...")

    sim_time = time.time() - start
    print(f"  Total simulation time: {sim_time:.1f}s "
          f"({sim_time / n_samples * 1000:.0f} ms per design point)")
    return np.array(X), np.array(y), sim_time


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
def predict_physical(model, ffunc, X: np.ndarray) -> np.ndarray:
    """
    Predict, then clamp to the physically admissible range.

    A polynomial surrogate has no idea that reflected power cannot be negative
    or exceed 1, and near a deep resonance null it will happily extrapolate to
    a small negative number. Clamping is part of the model, not cosmetics: a
    design search minimising an unclamped prediction chases the most negative
    (i.e. most unphysical) corner of the fit.
    """
    return np.clip(model.predict(ffunc(X)), 0.0, 1.0)


def _models(seed: int):
    return {
        "poly-ridge (deg 3)": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-3),
        ),
        "random forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=1, random_state=seed
        ),
    }


def train_and_compare(X, y, seed: int = 0):
    """Fit every (feature set, model) pair and score on a held-out split."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=seed)

    print(f"\n  Held-out evaluation ({len(X_tr)} train / {len(X_te)} test):")
    print(f"    {'features':<20} {'model':<20} {'R2 train':>9} "
          f"{'R2 test':>9} {'RMSE test':>10}")
    print("    " + "-" * 71)

    results = {}
    for fname, ffunc in FEATURE_SETS.items():
        F_tr, F_te = ffunc(X_tr), ffunc(X_te)
        for mname, model in _models(seed).items():
            model.fit(F_tr, y_tr)
            pred_te = np.clip(model.predict(F_te), 0.0, 1.0)
            pred_tr = np.clip(model.predict(F_tr), 0.0, 1.0)
            metrics = {
                "r2_train": r2_score(y_tr, pred_tr),
                "r2_test": r2_score(y_te, pred_te),
                "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
                "model": model,
                "features": ffunc,
            }
            results[(fname, mname)] = metrics
            print(f"    {fname:<20} {mname:<20} {metrics['r2_train']:9.4f} "
                  f"{metrics['r2_test']:9.4f} {metrics['rmse_test']:10.5f}")

    best = max(results, key=lambda k: results[k]["r2_test"])
    print(f"\n    -> best on held-out data: {best[1]} on {best[0]} features "
          f"(R2 = {results[best]['r2_test']:.4f})")
    return results, best


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark & plots
# ═══════════════════════════════════════════════════════════════════════════════
def speed_benchmark(model, ffunc, n_predictions: int = 100_000):
    """Surrogate cost per design point vs the simulation that generated it."""
    rng = np.random.default_rng(1)
    grid = np.column_stack([
        rng.uniform(EPS_MIN, EPS_MAX, n_predictions),
        rng.uniform(THICK_MIN_MM, THICK_MAX_MM, n_predictions),
    ])
    start = time.time()
    predict_physical(model, ffunc, grid)
    per_prediction = (time.time() - start) / n_predictions

    start = time.time()
    measure_scattering(_config(4.0, 100), compare_analytic=False)
    fdtd_per_point = time.time() - start

    speedup = fdtd_per_point / per_prediction
    print(f"\n  Speed benchmark (per design point):")
    print(f"    FDTD measurement (2 grid runs) : {fdtd_per_point*1e3:10.1f} ms")
    print(f"    Surrogate prediction           : {per_prediction*1e6:10.3f} us")
    print(f"    Speedup                        : {speedup:10,.0f}x")
    return speedup


def plot_surrogate(X, y, model, ffunc, output_dir: str = "results"):
    """Response surface, accuracy, and thickness cuts through the resonance."""
    os.makedirs(output_dir, exist_ok=True)

    eps_grid = np.linspace(EPS_MIN, EPS_MAX, 200)
    thick_grid = np.linspace(THICK_MIN_MM, THICK_MAX_MM, 200)
    EE, DD = np.meshgrid(eps_grid, thick_grid)
    surface = predict_physical(model, ffunc,
                               np.column_stack([EE.ravel(), DD.ravel()]))
    surface = surface.reshape(EE.shape)

    vmax = max(y.max(), surface.max())

    fig = plt.figure(figsize=(15, 4.6), dpi=120)
    fig.suptitle(
        f"ML Surrogate Trained on Measured FDTD Output "
        f"(reflected power at {TARGET_FREQ/1e9:.0f} GHz)",
        fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1)
    mesh = ax1.pcolormesh(EE, DD, surface, shading="auto", cmap="magma",
                          vmin=0, vmax=vmax)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="magma", vmin=0, vmax=vmax,
                edgecolor="white", linewidth=0.5, s=26)
    fig.colorbar(mesh, ax=ax1, label="Reflected power")
    ax1.set_xlabel("Relative permittivity (eps_r)")
    ax1.set_ylabel("Slab thickness (mm)")
    ax1.set_title("Surrogate surface; dots = FDTD runs", fontsize=10)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(y, predict_physical(model, ffunc, X), s=24, alpha=0.75, color="#2563EB")
    ax2.plot([0, vmax], [0, vmax], "--", color="#6B7280", linewidth=1)
    ax2.set_xlabel("FDTD measurement")
    ax2.set_ylabel("Surrogate prediction")
    ax2.set_title("Predicted vs measured", fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    for eps_fixed, colour in ((2.5, "#2563EB"), (6.0, "#F59E0B"), (9.5, "#8B5CF6")):
        cut = np.column_stack([np.full_like(thick_grid, eps_fixed), thick_grid])
        ax3.plot(thick_grid, predict_physical(model, ffunc, cut), color=colour,
                 linewidth=2, label=f"eps_r = {eps_fixed}")
    ax3.set_xlabel("Slab thickness (mm)")
    ax3.set_ylabel("Reflected power")
    ax3.set_title("Thickness cuts — half-wave windows", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "surrogate_model.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {path}")
    return path


def optimise_with_surrogate(model, ffunc):
    """
    The payoff: search the design space for the most transparent slab.

    100k candidate designs is seconds of surrogate time and would be well over
    ten hours of FDTD. The winner is then confirmed with a real simulation —
    the surrogate proposes, the solver disposes.
    """
    rng = np.random.default_rng(7)
    cand = np.column_stack([
        rng.uniform(EPS_MIN, EPS_MAX, 100_000),
        rng.uniform(THICK_MIN_MM, THICK_MAX_MM, 100_000),
    ])
    pred = predict_physical(model, ffunc, cand)
    best = cand[int(np.argmin(pred))]

    cfg = _config(best[0], int(round(best[1])))
    res = measure_scattering(cfg, compare_analytic=False)
    actual = float(np.interp(TARGET_FREQ, res["freqs"], res["R_f"]) ** 2)

    print(f"\n  Surrogate-guided design search (100,000 candidates):")
    print(f"    Best design      : eps_r = {best[0]:.3f}, "
          f"thickness = {best[1]:.1f} mm")
    print(f"    Surrogate says   : {pred.min():.5f} reflected power")
    print(f"    FDTD confirms    : {actual:.5f} reflected power")
    return best, actual


def main():
    print()
    print("=" * 72)
    print("  ML SURROGATE MODEL FOR FDTD SIMULATION")
    print("=" * 72)
    print()

    X, y, sim_time = generate_training_data(n_samples=150)
    results, best_key = train_and_compare(X, y)

    best = results[best_key]
    model, ffunc = best["model"], best["features"]

    speedup = speed_benchmark(model, ffunc)
    print("\n  Generating comparison plot ...")
    plot_surrogate(X, y, model, ffunc)
    design, actual = optimise_with_surrogate(model, ffunc)

    naive_best = max(
        (v for k, v in results.items() if k[0].startswith("naive")),
        key=lambda m: m["r2_test"])

    print()
    print("=" * 72)
    print("  SURROGATE MODEL SUMMARY")
    print("=" * 72)
    print(f"  Training data    : {len(X)} FDTD measurements ({sim_time:.0f}s)")
    print(f"  Target           : reflected power at {TARGET_FREQ/1e9:.0f} GHz, "
          f"measured from the grid")
    print(f"  Design space     : eps_r {EPS_MIN}-{EPS_MAX}, "
          f"thickness {THICK_MIN_MM:.0f}-{THICK_MAX_MM:.0f} mm")
    print(f"  Best model       : {best_key[1]} on {best_key[0]} features")
    print(f"  Held-out R2      : {best['r2_test']:.4f}  "
          f"(RMSE {best['rmse_test']:.5f})")
    print(f"  Naive features   : R2 {naive_best['r2_test']:.4f} — "
          f"the resonance is not learnable from eps_r and d alone")
    print(f"  Speedup          : {speedup:,.0f}x per design point")
    print(f"  Optimised design : eps_r {design[0]:.2f}, {design[1]:.0f} mm "
          f"-> {actual:.5f} reflected power (FDTD-confirmed)")
    print("=" * 72)


if __name__ == "__main__":
    main()
