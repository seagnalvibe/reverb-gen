import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reverberation_3d as r3d
import zone_map as zm


class TestZoneMap(unittest.TestCase):
    def test_presets(self):
        for d in zm.ZONES.values():
            for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
                self.assertIn(k, d)
        for p in ("toulon", "calvi"):
            self.assertIn(p, zm.POINTS)

    def test_build_cfg(self):
        class A:
            pass
        a = A()
        a.zone = "toulon_calvi"
        a.bbox = None
        a.outdir = None
        cfg, zone = zm.build_cfg(a)
        self.assertEqual(zone, "toulon_calvi")
        self.assertEqual(cfg["outdir"], "out_toulon_calvi")
        self.assertEqual(cfg["lon_max"], 9.0)
        a.zone = None
        a.bbox = (42.4, 43.3, 5.8, 9.0)
        a.outdir = "/tmp/x"
        cfg, zone = zm.build_cfg(a)
        self.assertEqual(zone, "custom")
        self.assertEqual((cfg["lat_min"], cfg["lon_max"], cfg["outdir"]),
                         (42.4, 9.0, "/tmp/x"))

    def test_run_and_display(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = r3d.copy.deepcopy(r3d.CONFIG)
            r3d.deep_update(cfg, {
                "lat_min": 42.0, "lat_max": 42.2, "dlat": 0.1,
                "lon_min": 3.0, "lon_max": 3.2, "dlon": 0.1,
                "depth_max": 100.0, "ddepth": 20.0,
                "freqs_khz": [0.1, 12.0, 1000.0],
                "hours_utc": [2, 14],
                "geotiff": {"freq_khz": 12.0, "hours_utc": [2, 14],
                            "depth_step_m": 50.0},
                "outdir": td})
            r3d.main(cfg)
            f = zm.load_nc(td)
            maps = zm.plot_maps(f, td, freq_khz=12.0, hours=(2, 14),
                                depths=(0.0, 100.0))
            self.assertTrue(maps)
            for p in maps:
                self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 5000, p)
            zm.POINTS["pA"] = (42.05, 3.05)
            zm.POINTS["pB"] = (42.15, 3.15)
            tr = zm.plot_transect(f, td, "pA", "pB", freq_khz=12.0, hour=14)
            self.assertTrue(os.path.exists(tr) and os.path.getsize(tr) > 5000)
            f.close()

    def test_transect_outside_grid(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = r3d.copy.deepcopy(r3d.CONFIG)
            r3d.deep_update(cfg, {
                "lat_min": 42.0, "lat_max": 42.2, "dlat": 0.1,
                "lon_min": 3.0, "lon_max": 3.2, "dlon": 0.1,
                "depth_max": 100.0, "ddepth": 20.0,
                "freqs_khz": [12.0], "hours_utc": [14],
                "geotiff": {"freq_khz": 12.0, "hours_utc": [14],
                            "depth_step_m": 50.0},
                "outdir": td})
            r3d.main(cfg)
            f = zm.load_nc(td)
            with self.assertRaises(ValueError):
                zm.plot_transect(f, td, "toulon", "calvi")
            f.close()


if __name__ == "__main__":
    unittest.main()
