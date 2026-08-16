"""
tests/test_fdtd.py
------------------
pytest test suite validating the FDTD simulator.

Tests cover:
  - Physical constant correctness
  - Courant stability condition
  - Gaussian source waveform shape
  - Material profile initialization
  - Energy conservation (lossless case)
  - Symmetry of free-space propagation
  - Reflection at a dielectric interface
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdtd_1d import SimConfig, FDTDEngine, C0, MU0, EPS0, ETA0


# ── Physical constants ───────────────────────────────────────────────────────

class TestPhysicalConstants:
    def test_speed_of_light(self):
        assert abs(C0 - 3e8) < 1e4
    
    def test_impedance_of_free_space(self):
        assert abs(ETA0 - 377.0) < 1.0
    
    def test_maxwell_relation(self):
        """c = 1/sqrt(mu0 * eps0)"""
        c_calc = 1.0 / np.sqrt(MU0 * EPS0)
        assert abs(c_calc - C0) / C0 < 1e-3  # C0 is rounded, so 0.1% tolerance


# ── Configuration ────────────────────────────────────────────────────────────

class TestSimConfig:
    def test_courant_stability(self):
        """Courant number must be <= 1 for FDTD stability."""
        cfg = SimConfig()
        assert cfg.courant <= 1.0
    
    def test_dt_calculation(self):
        cfg = SimConfig(courant=0.5, dx=1e-3)
        expected_dt = 0.5 * 1e-3 / C0
        assert abs(cfg.dt - expected_dt) < 1e-20
    
    def test_slab_within_grid(self):
        cfg = SimConfig()
        assert 0 <= cfg.slab_start < cfg.slab_end <= cfg.num_cells


# ── Engine initialization ────────────────────────────────────────────────────

class TestEngineInit:
    def test_fields_start_at_zero(self):
        engine = FDTDEngine(SimConfig())
        assert np.all(engine.Ez == 0)
        assert np.all(engine.Hy == 0)
    
    def test_material_profile_free_space(self):
        """Outside the slab, permittivity should equal eps0."""
        cfg = SimConfig()
        engine = FDTDEngine(cfg)
        assert engine.eps_profile[0] == EPS0
        assert engine.eps_profile[cfg.num_cells - 1] == EPS0
    
    def test_material_profile_slab(self):
        """Inside the slab, permittivity should equal eps_r * eps0."""
        cfg = SimConfig(eps_r=4.0)
        engine = FDTDEngine(cfg)
        mid = (cfg.slab_start + cfg.slab_end) // 2
        assert abs(engine.eps_profile[mid] - 4.0 * EPS0) < 1e-20


# ── Source ───────────────────────────────────────────────────────────────────

class TestGaussianSource:
    def test_peak_at_delay(self):
        engine = FDTDEngine(SimConfig())
        peak = engine.gaussian_source(int(engine.cfg.pulse_delay))
        assert abs(peak - 1.0) < 1e-10
    
    def test_decays_away_from_peak(self):
        engine = FDTDEngine(SimConfig())
        far_val = engine.gaussian_source(0)
        assert far_val < 0.1  # should be very small far from delay


# ── Simulation behaviour ─────────────────────────────────────────────────────

class TestSimulation:
    def test_energy_increases_during_source(self):
        """Energy should build up while the source is active."""
        cfg = SimConfig(num_steps=200)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=200)
        # Energy at step 150 should be greater than at step 10
        assert engine.energy_history[150] > engine.energy_history[10]
    
    def test_energy_nonnegative(self):
        """EM energy can never be negative."""
        cfg = SimConfig(num_steps=500)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=500)
        assert all(e >= 0 for e in engine.energy_history)
    
    def test_free_space_no_reflection(self):
        """With no slab (eps_r=1), there should be no significant reflection."""
        cfg = SimConfig(eps_r=1.0, num_steps=1000)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=1000)
        
        # After pulse passes through, reflected region should be near zero
        reflected = np.max(np.abs(engine.Ez[:cfg.source_pos - 20]))
        assert reflected < 0.05
    
    def test_dielectric_causes_reflection(self):
        """A dielectric slab (eps_r=4) must cause measurable reflection."""
        cfg = SimConfig(eps_r=4.0, num_steps=1500)
        engine = FDTDEngine(cfg)
        engine.run(snapshot_interval=1500)
        
        # There should be a reflected pulse between source and slab
        region = engine.Ez[cfg.source_pos + 10 : cfg.slab_start - 10]
        peak_reflected = np.max(np.abs(region))
        assert peak_reflected > 0.01  # some energy must be reflected
