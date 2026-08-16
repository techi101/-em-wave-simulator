# 1-D FDTD Electromagnetic Wave Simulator

> A Python-based Finite-Difference Time-Domain (FDTD) simulator that models electromagnetic wave propagation through dielectric materials using Maxwell's equations on a Yee grid. Implements the same numerical method used by commercial EDA tools like Keysight ADS, Ansys HFSS, and CST Studio.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-green?style=flat-square)](https://numpy.org)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen?style=flat-square)](#running-tests)

---

## What It Does

This simulator sends a **Gaussian electromagnetic pulse** through a 1-D simulation domain containing a **dielectric slab** (e.g., FR-4 PCB substrate, glass, biological tissue). It demonstrates:

1. **Wave reflection and transmission** at material boundaries
2. **Partial energy reflection** when the wave hits a dielectric interface (Fresnel equations)
3. **Velocity reduction** inside the dielectric (wave slows down by factor of `1/sqrt(eps_r)`)
4. **Energy conservation tracking** to validate numerical stability

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the default simulation (FR-4 PCB substrate, eps_r=4.0)
python fdtd_1d.py

# Run with a custom material (e.g., glass, eps_r=7.0)
python fdtd_1d.py --eps-r 7.0

# Run with a lossy material (conductivity = 0.01 S/m)
python fdtd_1d.py --eps-r 4.0 --sigma 0.01

# Run the automated parameter sweep
python parameter_sweep.py

# Run the test suite
pytest tests/ -v
```

---

## Physics Implemented

### Maxwell's Curl Equations (Discretised on Yee Grid)

The FDTD method solves Maxwell's equations by discretising space and time on a staggered grid (the Yee grid), where E and H fields are offset by half a cell:

```
dEz/dt = (1/eps) * dHy/dx     (Faraday's Law)
dHy/dt = (1/mu)  * dEz/dx     (Ampère's Law)
```

These are converted to update equations:
```
Hy[i]^{n+1/2} = Hy[i]^{n-1/2} + (dt/(mu*dx)) * (Ez[i+1]^n - Ez[i]^n)
Ez[i]^{n+1}   = Ca * Ez[i]^n  + Cb * (Hy[i]^{n+1/2} - Hy[i-1]^{n+1/2})
```

Where `Ca` and `Cb` are material-dependent update coefficients that account for permittivity and conductivity.

### Key Numerical Concepts
- **Courant Stability Condition:** `S = c*dt/dx ≤ 1` (must be satisfied or the simulation diverges)
- **Mur Absorbing Boundary Conditions:** Prevents artificial reflections from grid edges
- **Gaussian Pulse Source:** Broadband excitation with smooth spectral content

---

## Project Structure

```
em-wave-simulator/
├── fdtd_1d.py           ← Core FDTD engine + visualization + report generator
├── parameter_sweep.py   ← Automated design study (sweep eps_r from 1 to 9)
├── tests/
│   └── test_fdtd.py     ← 15 pytest tests (physics, stability, behaviour)
├── results/             ← Auto-generated plots and reports
│   ├── wave_propagation.png
│   ├── energy_conservation.png
│   ├── material_profile.png
│   ├── parameter_sweep.png
│   └── simulation_report.txt
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why 1-D instead of 2-D/3-D?**
1-D is sufficient to demonstrate the core FDTD algorithm (Yee updates, stability, ABCs, material interfaces) without the complexity of a 2-D/3-D mesh. The update equations are mathematically identical — extending to higher dimensions adds more field components but the same algorithm applies.

**Why Mur ABC instead of PML?**
First-order Mur ABCs are analytically derived and transparent to explain. Perfectly Matched Layers (PML) are more accurate but add significant implementation complexity. For a 1-D proof of concept, Mur is sufficient and pedagogically clearer.

**Why NumPy instead of a simulation framework?**
Building the solver from scratch with NumPy demonstrates understanding of the underlying physics and numerical methods — not just the ability to use a black-box tool.

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_fdtd.py::TestPhysicalConstants::test_speed_of_light           PASSED
tests/test_fdtd.py::TestPhysicalConstants::test_impedance_of_free_space  PASSED
tests/test_fdtd.py::TestPhysicalConstants::test_maxwell_relation         PASSED
tests/test_fdtd.py::TestSimConfig::test_courant_stability                PASSED
tests/test_fdtd.py::TestSimConfig::test_dt_calculation                   PASSED
tests/test_fdtd.py::TestSimConfig::test_slab_within_grid                 PASSED
tests/test_fdtd.py::TestEngineInit::test_fields_start_at_zero            PASSED
tests/test_fdtd.py::TestEngineInit::test_material_profile_free_space     PASSED
tests/test_fdtd.py::TestEngineInit::test_material_profile_slab           PASSED
tests/test_fdtd.py::TestGaussianSource::test_peak_at_delay               PASSED
tests/test_fdtd.py::TestGaussianSource::test_decays_away_from_peak       PASSED
tests/test_fdtd.py::TestSimulation::test_energy_increases_during_source  PASSED
tests/test_fdtd.py::TestSimulation::test_energy_nonnegative              PASSED
tests/test_fdtd.py::TestSimulation::test_free_space_no_reflection        PASSED
tests/test_fdtd.py::TestSimulation::test_dielectric_causes_reflection    PASSED

15 passed in 0.79s
```

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.10+ |
| Scientific Computing | NumPy |
| Visualization | Matplotlib |
| Testing | pytest (15 tests) |
| CLI | argparse (stdlib) |
| No external simulation tools required | Pure Python + NumPy |
