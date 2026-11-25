#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genere la carte S_v 3D d'une zone puis l'affiche, en reutilisant
reverberation_3d.py :
  - cartes horizontales S_v (tranches profondeur x heure, isobathes en
    overlay) -> outdir/figures/map_*.png
  - coupe verticale S_v le long d'un transect entre deux points nommes
    -> outdir/figures/transect_*.png

Zones pre definies : golfe_du_lion (demo), toulon_calvi.
Points nommes : toulon, calvi, marseille, nice, ajaccio, banyuls.

Usage:
  python zone_map.py --zone toulon_calvi --bathy auto --chl emodnet
  python zone_map.py --zone golfe_du_lion --freqs 0.1,12,1000 --hours 2,14
  python zone_map.py --zone toulon_calvi --no-run --transect toulon calvi
  python zone_map.py --bbox 42.4 43.3 5.8 9.0 --transect toulon calvi \
      --slice-depths 0,120,400 --slice-freq 12
"""

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reverberation_3d as r3d

ZONES = {
    "golfe_du_lion": {"lat_min": 42.0, "lat_max": 43.2,
                      "lon_min": 3.0, "lon_max": 4.8},
    "toulon_calvi": {"lat_min": 42.4, "lat_max": 43.3,
                     "lon_min": 5.8, "lon_max": 9.0},
}

POINTS = {
    "toulon": (43.12, 5.93),
    "calvi": (42.57, 8.76),
    "marseille": (43.30, 5.37),
    "nice": (43.70, 7.27),
    "ajaccio": (41.92, 8.74),
    "banyuls": (42.48, 3.13),
}


def build_cfg(args):
    """Construit la config a partir de --zone / --bbox / --outdir."""
    cfg = copy.deepcopy(r3d.CONFIG)
    zone = args.zone
    if zone:
        if zone not in ZONES:
            raise ValueError("zone inconnue : %s (zones : %s)"
                             % (zone, sorted(ZONES)))
        r3d.deep_update(cfg, ZONES[zone])
    if args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        cfg.update(lat_min=lat_min, lat_max=lat_max,
                   lon_min=lon_min, lon_max=lon_max)
        zone = zone or "custom"
    if args.outdir:
        cfg["outdir"] = args.outdir
    elif zone:
        cfg["outdir"] = "out_" + zone
    return cfg, zone


def parse_freqs(args):
    if args.freqs_log:
        kmin, kmax, n = args.freqs_log
        return list(np.logspace(np.log10(kmin), np.log10(kmax), int(n)))
    if args.freqs:
        return [float(x) for x in args.freqs.split(",")]
    return None


def _nearest(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr, float) - v)))


def load_nc(outdir):
    """Ouvre outdir/reverb_volume_3d.nc (scipy netcdf_file)."""
    from scipy.io import netcdf_file
    path = os.path.join(outdir, "reverb_volume_3d.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(path + " (lancer d'abord sans --no-run)")
    return netcdf_file(path, mmap=False)


def plot_maps(f, outdir, freq_khz=12.0, hours=(2, 14), depths=(0.0, 120.0, 400.0)):
    """Cartes horizontales S_v par (heure, profondeur) -> liste de PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = f.variables["lat"][:]
    lon = f.variables["lon"][:]
    z = f.variables["depth"][:]
    fr = f.variables["frequency"][:]
    t = f.variables["time"][:]
    sv = f.variables["Sv"][:]
    ifr = _nearest(fr, freq_khz)
    la2, lo2 = np.meshgrid(lat, lon, indexing="ij")
    hb = f.variables["bathymetry"][:] if "bathymetry" in f.variables else None
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    paths = []
    for hu in hours:
        it = _nearest(t, hu)
        for zd in depths:
            iz = _nearest(z, zd)
            sl = sv[it, ifr, iz]
            if np.all(np.isnan(sl)):
                continue
            vmin, vmax = np.nanpercentile(sl, [2.0, 98.0])
            fig, ax = plt.subplots(figsize=(8, 6))
            m = ax.imshow(sl, origin="lower",
                          extent=[lon[0], lon[-1], lat[0], lat[-1]],
                          aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            if hb is not None:
                ax.contour(lo2, la2, hb, levels=[200.0, 1000.0],
                           colors="k", linewidths=0.8)
            ax.set_xlabel("lon (deg E)")
            ax.set_ylabel("lat (deg N)")
            ax.set_title("S_v %g kHz, %d m, %02dh UTC" % (fr[ifr], z[iz], t[it]))
            fig.colorbar(m, ax=ax, label="S_v (dB re 1 m^-1)")
            fig.tight_layout()
            path = os.path.join(figdir, "map_f%gkHz_z%04dm_%02dhUTC.png"
                                % (fr[ifr], z[iz], t[it]))
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_transect(f, outdir, name0, name1, freq_khz=12.0, hour=14, n=300):
    """Coupe verticale S_v le long du transect name0 -> name1 -> PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator
    for nm in (name0, name1):
        if nm not in POINTS:
            raise ValueError("point inconnu : %s (points : %s)"
                             % (nm, sorted(POINTS)))
    lat = f.variables["lat"][:]
    lon = f.variables["lon"][:]
    z = f.variables["depth"][:]
    fr = f.variables["frequency"][:]
    t = f.variables["time"][:]
    sv = f.variables["Sv"][:]
    ifr = _nearest(fr, freq_khz)
    it = _nearest(t, hour)
    lat0, lon0 = POINTS[name0]
    lat1, lon1 = POINTS[name1]
    if not (float(lat.min()) <= min(lat0, lat1) and max(lat0, lat1) <= float(lat.max())
            and float(lon.min()) <= min(lon0, lon1) and max(lon0, lon1) <= float(lon.max())):
        raise ValueError("transect %s-%s hors de la grille [%.3f,%.3f]x[%.3f,%.3f] : "
                         "agrandir la zone (--zone/--bbox)"
                         % (name0, name1, lat.min(), lat.max(), lon.min(), lon.max()))
    la = np.linspace(lat0, lat1, n)
    lo = np.linspace(lon0, lon1, n)
    x_km = np.hypot((lo - lon0) * 111.32 * np.cos(np.radians(la)),
                    (la - lat0) * 110.57)
    pts = np.stack([la, lo], -1)
    section = np.full((len(z), n), np.nan)
    for k in range(len(z)):
        plane = sv[it, ifr, k]
        section[k] = RegularGridInterpolator(
            (lat, lon), plane, bounds_error=False, fill_value=np.nan)(pts)
    vmin, vmax = np.nanpercentile(section, [2.0, 98.0])
    fig, ax = plt.subplots(figsize=(11, 5))
    m = ax.pcolormesh(x_km, -z, section, shading="auto", cmap="viridis",
                     vmin=vmin, vmax=vmax)
    if "bathymetry" in f.variables:
        hb = f.variables["bathymetry"][:]
        h_tr = RegularGridInterpolator((lat, lon), hb, bounds_error=False,
                                       fill_value=np.nan)(pts)
        ax.plot(x_km, -h_tr, "k-", lw=1.5)
    ax.set_xlabel("distance %s -> %s (km)" % (name0, name1))
    ax.set_ylabel("immersion (m)")
    ax.set_title("S_v %g kHz, %02dh UTC — coupe %s-%s"
                 % (fr[ifr], t[it], name0, name1))
    fig.colorbar(m, label="S_v (dB re 1 m^-1)")
    fig.tight_layout()
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "transect_%s_%s_f%gkHz_%02dhUTC.png"
                        % (name0, name1, fr[ifr], t[it]))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", choices=sorted(ZONES), default=None,
                    help="zone pre definie (golfe_du_lion, toulon_calvi)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--modele", choices=["apl_uw", "searay"], default="apl_uw")
    ap.add_argument("--chl", default=None, metavar="emodnet|erddap|FICHIER")
    ap.add_argument("--bathy", default=None, metavar="FICHIER|auto")
    ap.add_argument("--freqs", default=None, metavar="K1,K2,...")
    ap.add_argument("--freqs-log", nargs=3, type=float, default=None,
                    metavar=("KMIN", "KMAX", "N"))
    ap.add_argument("--hours", default=None, metavar="H1,H2,...")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--no-run", action="store_true",
                    help="ne genere pas le netCDF, relit celui de --outdir")
    ap.add_argument("--transect", nargs=2, action="append", default=None,
                    metavar=("P0", "P1"),
                    help="points nommes (voir POINTS), ex. --transect toulon calvi")
    ap.add_argument("--slice-freq", type=float, default=12.0,
                    help="frequence des cartes/coupe (kHz, la plus proche)")
    ap.add_argument("--slice-hours", default="2,14", help="heures des cartes, ex. '2,14'")
    ap.add_argument("--slice-depths", default="0,120,400",
                    help="profondeurs des cartes (m), ex. '0,120,400'")
    ap.add_argument("--transect-hour", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="timeout des telechargements directs (s)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--data-dir", default=None,
                    help="dossier de cache prioritaire pour les CSV/TIF telecharges")
    ap.add_argument("--offline", action="store_true",
                    help="aucun appel reseau : exige les fichiers locaux")
    args = ap.parse_args(argv)

    cfg, zone = build_cfg(args)
    if not args.no_run:
        freqs = parse_freqs(args)
        hours = [int(h) for h in args.hours.split(",")] if args.hours else None
        chl_src = args.chl or cfg["chlorophyll"]["source"]
        if chl_src == "none":
            chl_src = None
        bathy_src = args.bathy or cfg["bathymetry"]["source"]
        r3d.main(cfg, args.modele, chl_src=chl_src, bathy_src=bathy_src,
                 freqs=freqs, hours=hours, date_str=args.date,
                 timeout=args.timeout, retries=args.retries,
                 data_dir=args.data_dir, offline=args.offline)

    f = load_nc(cfg["outdir"])
    hours = [int(h) for h in args.slice_hours.split(",")]
    depths = [float(d) for d in args.slice_depths.split(",")]
    for p in plot_maps(f, cfg["outdir"], freq_khz=args.slice_freq,
                       hours=hours, depths=depths):
        print("carte :", p)
    trans = args.transect or ([("toulon", "calvi")] if zone == "toulon_calvi" else [])
    for p0, p1 in trans:
        print("coupe :", plot_transect(f, cfg["outdir"], p0, p1,
                                       freq_khz=args.slice_freq,
                                       hour=args.transect_hour))
    f.close()


if __name__ == "__main__":
    main()
