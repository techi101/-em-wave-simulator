"""
tests/test_fdtd.py
------------------
pytest test suite validating the FDTD simulator.

The tests are organised around claims that can actually fail:

  - the absorbing boundaries absorb (regression for a detuned Mur ABC)
  - a lossless slab conserves energy: R + T = 1
  - measured R(f) and T(f) match the closed-form slab solution
  - the measurement does not depend on how long the simulation ran
  - the scheme converges at second order under grid refinement
  - a lossy slab absorbs, and free space scatters nothing

An earlier version of this suite asserted things like "peak |Ez| left of the
slab is > 0.01 after 1500 steps", which measured leftover numerical residue
rather than physics and passed regardless of whether the solver was correct.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdtd_1d import C0, EPS0, ETA0, MU0, FDTDEngine, SimConfig
from scattering import analytic_slab_rt, default_num_steps, measure_scattering


# One recording length for every default-geometry measurement in this file,
# sized for the slowest case here (eps_r = 9). scattering.py caches the
# empty-grid reference run per (geometry, num_steps), so holding num_steps
# fixed means that run happens once for the whole suite instead of once per
# eps_r. Same physics, roughly half the wall clock.
SHARED_STEPS = default_num_steps(SimConfig(eps_r=9.0))


# ── Shared, cached measurements (each is two full grid runs) ─────────────────

@pytest.fixture(scope="module")
def slab_measurement():
    return measure_scattering(SimConfig(eps_r=4.0), num_steps=SHARED_STEPS)


@pytest.fixture(scope="module")
def empty_engine():
    cfg = SimConfig(eps_r=1.0, num_steps=6000)
    return FDTDEngine(cfg).run(snapshot_interval=10 ** 9)


# ── Physical constants ───────────────────────────────────────────────────────

class TestPhysicalConstants:
    def test_speed_of_light(self):
        assert abs(C0 - 2.99792458e8) < 1e3

    def test_impedance_of_free_space(self):
        assert abs(ETA0 - 376.73) < 0.1

    def test_maxwell_relation_is_exact(self):
        """
        c must be DERIVED from mu0 and eps0, not hard-coded to 3e8.

        The Mur ABC coefficient and the Courant number both compare c*dt to
        dx. A c that disagrees with the mu0/eps0 in the update coefficients
        detunes the boundary, so this is a correctness constraint rather than
        a rounding preference.
        """
        assert C0 == pytest.approx(1.0 / np.sqrt(MU0 * EPS0), rel=1e-15)


# ── Configuration ────────────────────────────────────────────────────────────

class TestSimConfig:
    def test_courant_stability(self):
        assert SimConfig().courant <= 1.0

    def test_dt_calculation(self):
        cfg = SimConfig(courant=0.5, dx=1e-3)
        assert cfg.dt == pytest.approx(0.5 * 1e-3 / C0, rel=1e-12)

    def test_slab_within_grid(self):
        cfg = SimConfig()
        assert 0 <= cfg.slab_start < cfg.slab_end <= cfg.num_cells

    def test_probes_default_between_source_slab_and_boundary(self):
        cfg = SimConfig()
        assert cfg.source_pos < cfg.probe_refl < cfg.slab_start
        assert cfg.slab_end < cfg.probe_trans < cfg.num_cells

    def test_probe_outside_grid_is_rejected(self):
        with pytest.raises(ValueError):
            SimConfig(num_cells=600, probe_refl=900)

    def test_slab_outside_grid_is_rejected(self):
        """
        Numpy slicing clips silently, so a slab placed past the end of the
        grid used to yield a simulation containing no slab at all — and a
        report that looked entirely plausible.
        """
        with pytest.raises(ValueError, match="does not fit"):
            SimConfig(num_cells=200)

    def test_unstable_courant_is_rejected(self):
        with pytest.raises(ValueError, match="stability"):
            SimConfig(courant=1.5)

    def test_probe_ordering_is_enforced(self):
        """The measurement is only meaningful for source < probe < slab < probe."""
        with pytest.raises(ValueError, match="strictly left"):
            SimConfig(probe_refl=300)          # would sit inside the slab
        with pytest.raises(ValueError, match="strictly left"):
            SimConfig(probe_trans=300)         # would sit inside the slab


# ── Engine initialisation ────────────────────────────────────────────────────

class TestEngineInit:
    def test_fields_start_at_zero(self):
        engine = FDTDEngine(SimConfig())
        assert np.all(engine.Ez == 0)
        assert np.all(engine.Hy == 0)

    def test_material_profile_free_space(self):
        cfg = SimConfig()
        engine = FDTDEngine(cfg)
        assert engine.eps_profile[0] == EPS0
        assert engine.eps_profile[cfg.num_cells - 1] == EPS0

    def test_material_profile_slab(self):
        cfg = SimConfig(eps_r=4.0)
        engine = FDTDEngine(cfg)
        mid = (cfg.slab_start + cfg.slab_end) // 2
        assert engine.eps_profile[mid] == pytest.approx(4.0 * EPS0, rel=1e-12)

    def test_probes_record_every_step(self):
        cfg = SimConfig(num_steps=300)
        engine = FDTDEngine(cfg).run(snapshot_interval=10 ** 9)
        for cell in (cfg.probe_refl, cfg.probe_trans):
            assert engine.probe_data[cell].shape == (300,)
        assert np.any(engine.probe_data[cfg.probe_refl] != 0)


# ── Source ───────────────────────────────────────────────────────────────────

class TestGaussianSource:
    def test_peak_at_delay(self):
        engine = FDTDEngine(SimConfig())
        assert engine.gaussian_source(int(engine.cfg.pulse_delay)) == pytest.approx(1.0)

    def test_decays_away_from_peak(self):
        engine = FDTDEngine(SimConfig())
        assert engine.gaussian_source(0) < 0.1


# ── Numerical stability ──────────────────────────────────────────────────────

class TestStability:
    def test_fields_stay_bounded(self):
        """A violated Courant condition blows up exponentially; this must not."""
        engine = FDTDEngine(SimConfig(num_steps=3000)).run(snapshot_interval=10 ** 9)
        assert np.all(np.isfinite(engine.Ez))
        assert np.max(np.abs(engine.Ez)) < 10.0

    def test_energy_is_never_negative(self):
        engine = FDTDEngine(SimConfig(num_steps=500)).run(snapshot_interval=10 ** 9)
        assert all(e >= 0 for e in engine.energy_history)

    def test_energy_builds_while_the_source_is_active(self):
        engine = FDTDEngine(SimConfig(num_steps=200)).run(snapshot_interval=10 ** 9)
        assert engine.energy_history[150] > engine.energy_history[10]


# ── Absorbing boundaries ─────────────────────────────────────────────────────

class TestAbsorbingBoundaries:
    def test_empty_grid_echo_is_small(self, empty_engine):
        """
        Regression. update_E used to write Ez[1:], which includes the last
        cell, leaving the right-hand Mur ABC differencing a time-(n+1) value
        against a time-n one. That turned the right boundary into a ~90%
        mirror while the left boundary (where Ez[0] was already excluded)
        stayed correct. An empty grid must return essentially nothing.
        """
        cfg = empty_engine.cfg
        trace = empty_engine.probe_data[cfg.probe_refl]
        incident = np.max(np.abs(trace[:400]))
        echo = np.max(np.abs(trace[400:]))
        assert 20 * np.log10(echo / incident) < -40.0

    def test_empty_grid_drains(self, empty_engine):
        peak = max(empty_engine.energy_history)
        assert empty_engine.energy_history[-1] / peak < 1e-3

    def test_free_space_scatters_nothing(self):
        """With eps_r = 1 there is no slab, so R ~ 0 and T ~ 1."""
        result = measure_scattering(SimConfig(eps_r=1.0), num_steps=SHARED_STEPS)
        assert result["R_power"] < 1e-3
        assert result["T_power"] == pytest.approx(1.0, abs=1e-3)


# ── Energy conservation ──────────────────────────────────────────────────────

class TestEnergyConservation:
    @pytest.mark.parametrize("eps_r", [2.0, 4.0, 9.0])
    def test_lossless_slab_conserves_power(self, eps_r):
        """R + T = 1 exactly for a lossless slab. Nothing else is acceptable."""
        result = measure_scattering(SimConfig(eps_r=eps_r), num_steps=SHARED_STEPS)
        assert result["power_sum"] == pytest.approx(1.0, abs=2e-3)

    def test_lossy_slab_absorbs(self):
        lossless = measure_scattering(SimConfig(eps_r=4.0, sigma=0.0), num_steps=SHARED_STEPS)
        lossy = measure_scattering(SimConfig(eps_r=4.0, sigma=0.05), num_steps=SHARED_STEPS)
        assert lossy["power_sum"] < lossless["power_sum"] - 0.01
        assert lossy["T_power"] < lossless["T_power"]

    @pytest.mark.parametrize("sigma", [0.01, 0.05, 0.2])
    def test_lossy_slab_matches_analytic(self, sigma):
        """
        Pins the sign convention of the complex permittivity.

        With eps_c = eps_r + i*sigma/(omega*eps0) and Im(n) >= 0 the wave
        decays inside the slab; get the branch wrong and the closed form
        predicts gain instead. Only an FDTD comparison catches that, since
        "R + T < 1" alone is satisfied by either sign of the error.
        """
        result = measure_scattering(SimConfig(eps_r=4.0, sigma=sigma), num_steps=SHARED_STEPS)
        assert result["R_power"] == pytest.approx(
            result["R_power_analytic"], abs=5e-3)
        assert result["T_power"] == pytest.approx(
            result["T_power_analytic"], abs=5e-3)
        assert 0.0 < 1.0 - result["power_sum"] < 1.0      # genuine absorption

    def test_absorption_increases_with_conductivity(self):
        absorbed = [1.0 - measure_scattering(
            SimConfig(eps_r=4.0, sigma=s), num_steps=SHARED_STEPS)["power_sum"]
            for s in (0.0, 0.01, 0.05)]
        assert absorbed[0] < absorbed[1] < absorbed[2]


# ── Validation against the closed-form slab solution ─────────────────────────

class TestAnalyticSolution:
    def test_analytic_is_lossless(self):
        freqs = np.linspace(0.1e9, 8e9, 400)
        r, t = analytic_slab_rt(freqs, eps_r=4.0, thickness=0.1)
        assert np.allclose(np.abs(r) ** 2 + np.abs(t) ** 2, 1.0, atol=1e-12)

    def test_half_wave_window_is_transparent(self):
        """
        A slab an integer number of half-wavelengths thick vanishes: the two
        interface reflections cancel exactly.
        """
        eps_r, thickness = 4.0, 0.1          # n = 2, d = 100 mm
        n = np.sqrt(eps_r)
        f_half_wave = C0 / (2.0 * n * thickness)     # one half-wave in the slab
        r, t = analytic_slab_rt(np.array([f_half_wave]), eps_r, thickness)
        assert abs(r[0]) == pytest.approx(0.0, abs=1e-12)
        assert abs(t[0]) == pytest.approx(1.0, abs=1e-12)

    def test_power_average_equals_the_incoherent_slab_reflectance(self):
        """
        Averaged over many Fabry-Perot periods, the slab's power reflectance
        tends to the classic incoherent result

            R_incoherent = 2*R1 / (1 + R1),    R1 = |(1-n)/(1+n)|^2

        which for eps_r = 4 is exactly 0.2. Note that this is NOT the
        single-interface value R1 = 1/9: the back face contributes too. It is
        also the number the broadband FDTD measurement reports (~20.2%), so
        this ties the closed-form solution, the textbook identity, and the
        solver together.
        """
        eps_r = 4.0
        n = np.sqrt(eps_r)
        R1 = abs((1 - n) / (1 + n)) ** 2
        expected = 2 * R1 / (1 + R1)
        assert expected == pytest.approx(0.2, abs=1e-12)

        freqs = np.linspace(1e9, 9e9, 4001)
        r, _ = analytic_slab_rt(freqs, eps_r, thickness=0.1)
        assert np.mean(np.abs(r) ** 2) == pytest.approx(expected, rel=0.02)

    def test_fdtd_matches_analytic_spectrum(self, slab_measurement):
        assert slab_measurement["max_R_error"] < 0.10
        assert slab_measurement["rms_R_error"] < 0.03
        assert slab_measurement["max_T_error"] < 0.05

    def test_fdtd_matches_analytic_band_power(self, slab_measurement):
        assert slab_measurement["R_power"] == pytest.approx(
            slab_measurement["R_power_analytic"], abs=5e-3)
        assert slab_measurement["T_power"] == pytest.approx(
            slab_measurement["T_power_analytic"], abs=5e-3)

    def test_low_frequency_end_is_most_accurate(self, slab_measurement):
        """
        Error should be dominated by the high-frequency edge of the band,
        where there are fewest cells per wavelength. If it were not, the
        disagreement would be a bug rather than numerical dispersion.
        """
        err = np.abs(slab_measurement["R_f"] - slab_measurement["R_analytic_f"])
        half = len(err) // 2
        assert err[:half].mean() < err[half:].mean()


# ── Properties of the measurement itself ─────────────────────────────────────

class TestMeasurementRobustness:
    def test_result_does_not_depend_on_run_length(self):
        """
        Regression, and the central one. R and T used to be read off the final
        field snapshot, so the "measurement" decayed towards zero the longer
        you ran and was really a function of --steps. A physical coefficient
        cannot depend on how long you watched.
        """
        cfg = SimConfig(eps_r=4.0)
        short = measure_scattering(cfg, num_steps=9000)
        long = measure_scattering(cfg, num_steps=15000)
        assert short["R_power"] == pytest.approx(long["R_power"], abs=2e-3)
        assert short["T_power"] == pytest.approx(long["T_power"], abs=2e-3)

    def test_reflection_increases_with_contrast(self):
        r = [measure_scattering(SimConfig(eps_r=e), num_steps=SHARED_STEPS)["R_power"]
             for e in (1.5, 3.0, 6.0, 9.0)]
        assert all(a < b for a, b in zip(r, r[1:]))

    @pytest.mark.slow
    def test_second_order_convergence(self):
        """
        Halving dx should cut the error roughly fourfold for the second-order
        Yee scheme. A first-order error (a mis-centred update or boundary)
        would show ~2x instead.
        """
        def rms_error(k):
            cfg = SimConfig(num_cells=500 * k, dx=1e-3 / k, source_pos=100 * k,
                            slab_start=250 * k, slab_end=350 * k,
                            pulse_width=30.0 * k, pulse_delay=80.0 * k,
                            eps_r=4.0)
            return measure_scattering(cfg)["rms_R_error"]

        coarse, fine = rms_error(2), rms_error(4)
        assert fine < coarse
        assert coarse / fine > 3.0

    def test_default_run_length_drains_the_grid(self):
        cfg = SimConfig(eps_r=4.0)
        result = measure_scattering(cfg, num_steps=default_num_steps(cfg))
        assert result["residual_energy_fraction"] < 1e-3

    def test_coarse_frequency_resolution_is_rejected(self):
        """
        A too-short recording gives FFT bins so wide that the first one inside
        the band sits far above the band's lower edge, which shifts the band
        average for reasons that have nothing to do with the slab. Better to
        refuse than to return a number that quietly depends on run length.
        """
        with pytest.raises(RuntimeError, match="too coarse"):
            measure_scattering(SimConfig(eps_r=4.0), num_steps=2000)

    def test_band_lower_edge_is_pinned_to_a_frequency(self):
        """
        Regression. The band used to start at the first non-zero FFT bin,
        which is 1/(N*dt) and therefore moved with run length, so the reported
        R crept from 0.2059 at 4k steps to 0.2013 at 24k. The disagreement
        with theory was constant at +4e-4 throughout — the metric was moving,
        not the physics.
        """
        for steps in (8000, 16000):
            result = measure_scattering(SimConfig(eps_r=4.0), num_steps=steps)
            assert result["band_hz"][0] >= 0.2e9
            # Measured and analytic must track each other regardless of length.
            assert result["R_power"] - result["R_power_analytic"] == \
                pytest.approx(4.3e-4, abs=2e-4)

    def test_band_excludes_under_resolved_frequencies(self, slab_measurement):
        """The band must stop before numerical dispersion dominates."""
        cfg = SimConfig(eps_r=4.0)
        f_max = C0 / (20.0 * cfg.dx * np.sqrt(cfg.eps_r))
        assert slab_measurement["band_hz"][1] <= f_max * 1.001
