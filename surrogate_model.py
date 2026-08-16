"""
surrogate_model.py
------------------
Trains a Machine Learning surrogate model on FDTD simulation data.

Instead of running the full physics simulation every time (which can take
hours in 3-D), a surrogate model learns the input→output mapping and can
predict results in milliseconds.

This demonstrates:
  - Physics-informed machine learning
  - Surrogate modeling for engineering design acceleration
  - Automated data generation from simulation
  - Model evaluation and benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import os
import time

from fdtd_1d import SimConfig, FDTDEngine, compute_reflection_transmission


def generate_training_data(n_samples: int = 60):
    """
    Run FDTD simulations with varying eps_r to generate training data.
    
    In a real engineering workflow, this step could take hours (3-D sims).
    The surrogate model learns from this data so future predictions are instant.
    """
    print(f"  Generating {n_samples} FDTD simulations ...")
    
    # Sample eps_r values from 1.0 to 12.0
    eps_r_values = np.linspace(1.1, 12.0, n_samples)
    
    X = []  # inputs: [eps_r]
    y_refl = []  # output: peak reflected amplitude
    y_trans = []  # output: peak transmitted amplitude
    
    start = time.time()
    for i, eps_r in enumerate(eps_r_values):
        cfg = SimConfig(eps_r=float(eps_r), num_steps=2000)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=2000)
        
        rt = compute_reflection_transmission(engine)
        X.append([eps_r, np.sqrt(eps_r)])  # features: eps_r and refractive index
        y_refl.append(abs(rt["R_analytical"]))
        y_trans.append(rt["peak_transmitted_numerical"])
        
        if (i + 1) % 10 == 0:
            print(f"    Completed {i+1}/{n_samples} simulations ...")
    
    sim_time = time.time() - start
    print(f"  Total simulation time: {sim_time:.1f}s")
    
    return np.array(X), np.array(y_refl), np.array(y_trans), sim_time


def train_surrogate(X, y, name="reflection"):
    """Train a polynomial regression surrogate model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Polynomial features (degree 3) to capture non-linear physics
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Ridge regression (regularized to prevent overfitting)
    model = Ridge(alpha=0.01)
    model.fit(X_train_poly, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n  Surrogate Model ({name}):")
    print(f"    R2 (train): {r2_train:.4f}")
    print(f"    R2 (test):  {r2_test:.4f}")
    print(f"    RMSE (test): {rmse_test:.6f}")
    
    return model, poly, {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
    }


def speed_benchmark(model, poly, n_predictions=10000):
    """Compare surrogate prediction speed vs full FDTD simulation."""
    # Surrogate: predict 10,000 values
    X_new = np.column_stack([
        np.linspace(1.1, 12.0, n_predictions),
        np.sqrt(np.linspace(1.1, 12.0, n_predictions))
    ])
    
    start = time.time()
    X_poly = poly.transform(X_new)
    predictions = model.predict(X_poly)
    surrogate_time = time.time() - start
    
    # FDTD: run just 1 simulation for comparison
    start = time.time()
    cfg = SimConfig(eps_r=4.0, num_steps=2000)
    engine = FDTDEngine(cfg)
    engine.run(snapshot_interval=2000)
    fdtd_time = time.time() - start
    
    speedup = (fdtd_time * n_predictions) / max(surrogate_time, 1e-9)
    
    print(f"\n  Speed Benchmark:")
    print(f"    FDTD (1 sim):           {fdtd_time:.3f}s")
    print(f"    Surrogate ({n_predictions:,} predictions): {surrogate_time:.6f}s")
    print(f"    Estimated speedup:      {speedup:,.0f}x faster")
    
    return speedup, surrogate_time, fdtd_time


def plot_surrogate_vs_physics(X, y_refl, y_trans, model_refl, model_trans,
                               poly, output_dir="results"):
    """Plot surrogate predictions vs actual FDTD results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Dense prediction curve
    eps_dense = np.linspace(1.1, 12.0, 200)
    X_dense = np.column_stack([eps_dense, np.sqrt(eps_dense)])
    X_dense_poly = poly.transform(X_dense)
    
    pred_refl = model_refl.predict(X_dense_poly)
    pred_trans = model_trans.predict(X_dense_poly)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle("ML Surrogate Model vs FDTD Physics Simulation",
                 fontsize=13, fontweight="bold")
    
    # Reflection
    ax1.scatter(X[:, 0], y_refl, color="#EF4444", s=30, alpha=0.7,
                label="FDTD Simulation (ground truth)", zorder=5)
    ax1.plot(eps_dense, pred_refl, color="#2563EB", linewidth=2,
             label="Surrogate Model (ML prediction)")
    ax1.set_xlabel("Relative Permittivity (eps_r)")
    ax1.set_ylabel("|Reflection Coefficient|")
    ax1.set_title("Reflection Coefficient")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Transmission
    ax2.scatter(X[:, 0], y_trans, color="#10B981", s=30, alpha=0.7,
                label="FDTD Simulation (ground truth)", zorder=5)
    ax2.plot(eps_dense, pred_trans, color="#8B5CF6", linewidth=2,
             label="Surrogate Model (ML prediction)")
    ax2.set_xlabel("Relative Permittivity (eps_r)")
    ax2.set_ylabel("Peak Transmitted Ez (V/m)")
    ax2.set_title("Transmitted Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "surrogate_model.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {path}")


def main():
    print()
    print("=" * 60)
    print("  ML SURROGATE MODEL FOR FDTD SIMULATION")
    print("=" * 60)
    print()
    
    # Step 1: Generate training data from FDTD simulations
    X, y_refl, y_trans, sim_time = generate_training_data(n_samples=60)
    
    # Step 2: Train surrogate models
    print("\n  Training surrogate models ...")
    model_refl, poly, metrics_refl = train_surrogate(X, y_refl, "Reflection")
    model_trans, _, metrics_trans = train_surrogate(X, y_trans, "Transmission")
    
    # Step 3: Speed benchmark
    speedup, surr_time, fdtd_time = speed_benchmark(model_refl, poly)
    
    # Step 4: Visualization
    print("\n  Generating comparison plot ...")
    plot_surrogate_vs_physics(X, y_refl, y_trans, model_refl, model_trans,
                              poly)
    
    # Summary
    print()
    print("=" * 60)
    print("  SURROGATE MODEL SUMMARY")
    print("=" * 60)
    print(f"  Training data:  {len(X)} FDTD simulations")
    print(f"  Reflection  R2: {metrics_refl['r2_test']:.4f}")
    print(f"  Transmission R2: {metrics_trans['r2_test']:.4f}")
    print(f"  Speedup:        {speedup:,.0f}x faster than full FDTD")
    print(f"  Use case:       Instant design exploration without re-running")
    print(f"                  the full physics simulation")
    print("=" * 60)


if __name__ == "__main__":
    main()
