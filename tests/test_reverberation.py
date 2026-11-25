import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reverberation_3d as r3d

trapz = getattr(np, "trapezoid", None) or np.trapz


class TestPhysics(unittest.TestCase):
    def test_sound_speed_reference(self):
        self.assertAlmostEqual(r3d.sound_speed(0.0, 35.0, 0.0), 1449.2, places=1)

    def test_absorption_across_100hz_1mhz(self):
        a_lo = r3d.absorption_fg(0.1, 13.0, 38.2, 0.0)
        a_mid = r3d.absorption_fg(12.0, 13.0, 38.2, 0.0)
        a_hi = r3d.absorption_fg(1000.0, 13.0, 38.2, 0.0)
        self.assertTrue(0.0 < a_lo < 0.01, a_lo)
        self.assertLess(a_lo, a_mid)
        self.assertLess(a_mid, a_hi)
        self.assertTrue(100.0 < a_hi < 400.0, a_hi)

    def test_validate_freqs(self):
        cfg = r3d.CONFIG
        self.assertEqual(r3d.validate_freqs([1000.0, 0.1], cfg), [0.1, 1000.0])
        with self.assertRaises(ValueError):
            r3d.validate_freqs([0.05], cfg)
        with self.assertRaises(ValueError):
            r3d.validate_freqs([2000.0], cfg)

    def test_bladder_spectrum_band(self):
        cfg = r3d.CONFIG
        depth = np.arange(0.0, 1000.0 + 1e-9, 10.0)
        w = r3d.bladder_spectrum([0.1, 12.0, 1000.0], depth, cfg)
        self.assertEqual(w.shape, (3, len(depth)))
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertTrue(np.all(w >= 0.0))
        # la resonance (a0 = 2.5 mm a 400 m ~ 8.3 kHz) domine les extremes de bande
        iz = int(np.argmin(np.abs(depth - cfg["z_dsl_day"])))
        self.assertGreater(w[1, iz], w[0, iz])
        self.assertGreater(w[1, iz], w[2, iz])


class TestSolarDsl(unittest.TestCase):
    def test_solar_elevation(self):
        d = r3d.dt.datetime(2026, 6, 21)
        self.assertGreater(r3d.solar_elevation(42.6, 3.9, d, 12), 60.0)
        self.assertLess(r3d.solar_elevation(42.6, 3.9, d, 0), 0.0)

    def test_dsl_layer_biomass_conservation(self):
        cfg = r3d.CONFIG
        depth = np.arange(0.0, 1000.0 + 1e-9, 10.0)
        target = cfg["dsl_sigma_m"] * np.sqrt(2 * np.pi)
        deep = r3d.dsl_layer(depth[None, None, :], np.array([[400.0]]),
                             np.array([[1500.0]]), cfg)[0, 0]
        shelf = r3d.dsl_layer(depth[None, None, :], np.array([[400.0]]),
                              np.array([[90.0]]), cfg)[0, 0]
        self.assertAlmostEqual(trapz(deep, depth), target, delta=0.15 * target)
        self.assertAlmostEqual(trapz(shelf, depth), target, delta=0.2 * target)
        # sur le plateau la couche est comprimee vers le haut
        self.assertLess(float(depth[np.argmax(shelf)]), float(depth[np.argmax(deep)]))


class TestGeotiff(unittest.TestCase):
    def test_north_up_orientation(self):
        from PIL import Image
        lat = np.array([42.0, 42.1, 42.2])
        lon = np.array([3.0, 3.1])
        sl = np.repeat(np.arange(3, dtype="f4")[:, None], 2, axis=1)  # ligne i = lat[i]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.tif")
            r3d.write_geotiff(path, sl[::-1], lat, lon)  # convention main() : ligne 0 = nord
            a = np.array(Image.open(path))
            self.assertEqual(float(a[0, 0]), float(sl[-1, 0]))
            self.assertEqual(float(a[-1, 0]), float(sl[0, 0]))
            raw = open(path, "rb").read()
            tp = struct.pack("<6d", 0.0, 0.0, 0.0, float(lon[0]), float(lat[-1]), 0.0)
            self.assertGreater(raw.find(tp), 0)


class TestChlorophyll(unittest.TestCase):
    def test_negative_chl_clipped(self):
        cfg = r3d.CONFIG
        depth = np.arange(0.0, 1000.0 + 1e-9, 10.0)
        chl = np.full((len(depth), 3, 4), -5.0)
        p3, p_col = r3d.productivity_from_chl(chl, depth, cfg)
        self.assertTrue(np.all(np.isfinite(p3)))
        self.assertGreaterEqual(float(p3.min()), cfg["chlorophyll"]["p_min"])

    def test_chlorophyll_grid_extrapolation(self):
        depths = np.array([0.0, 100.0, 600.0])
        lats = np.array([42.0, 43.0])
        lons = np.array([3.0, 4.0])
        chl = np.full((3, 2, 2), 0.5)
        chl[0, 0, 0] = -2.0  # valeur aberrante -> clippee a 0
        depth_m = np.arange(0.0, 1000.0 + 1e-9, 10.0)
        lat = np.array([42.0, 42.5, 43.0])
        lon = np.array([3.0, 3.5, 4.0])
        out = r3d.chlorophyll_grid(depths, lats, lons, chl, depth_m, lat, lon)
        self.assertEqual(out.shape, (len(depth_m), 3, 3))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreaterEqual(float(out.min()), 0.0)
        # sous 600 m : prolongation constante de la valeur a 600 m
        self.assertAlmostEqual(float(out[-1, 1, 1]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
