#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carte 3D (lat, lon, immersion) de la reverbération volumique S_v
pour sonar actif, modele type APL-UW (resonance de vessies natatoires,
migration nythemerale de la couche de diffusion profonde) + Urick RL.

Plage de frequences supportee : 100 Hz (0.1 kHz) a 1 MHz (1000 kHz).
  - a 100 Hz, la resonance des vessies couvre les grandes vessies
    (a_max = 30 mm) ; le fond biologique domine sinon ;
  - a 1 MHz, l'absorption Francois-Garrison (~200 dB/km) limite la portee.

Sorties:
  - netCDF (CF) : Sv(time, frequency, depth, lat, lon) en dB re 1 m^-1
    (+ variable chlorophyll si un champ est fourni)
  - GeoTIFF WGS84 : tranches horizontales par immersion / frequence / heure
  - PNG quicklook
  - utilitaire RL (Urick + absorption Francois-Garrison)

Champ de chlorophylle (source unique, pas de melange) :
  --chl emodnet     DIVAnd saisonnier Mediterranee (ERDDAP EMODnet,
                    profils 0-600 m) ; 'auto' reste accepte (= emodnet)
  --chl erddap      MODIS Aqua L3 (ERDDAP CoastWatch), climatologie du mois
                    (produit journalier echantillonne au pas 12), extension
                    verticale expo depuis la valeur de surface satellite
  --chl fichier.csv CSV deja telecharge (format reconnu automatiquement)
  (sans --chl : proxy synthetique fonde sur la bathymetrie)

Bathymetrie :
  --bathy auto      telecharge le DTM EMODnet (WCS public, maille ~115 m)
  --bathy fichier.tif  reutilise un GeoTIFF produit par --bathy auto
  (sans --bathy : bathymetrie synthetique de demonstration)

References:
  - Urick R.J., Principles of Underwater Sound, 3rd ed., McGraw-Hill 1983 :
    reverberation volumique (RL = SL - 2TL + Sv + 10 log V)
  - Jackson D.R. et al., APL-UW High-Frequency Ocean Environmental Acoustic
    Models Handbook, APL-UW TR 9407 / AEAS 9501, octobre 1994 :
    modeles HF environnementaux (celerite, absorption, surface, fond).
    NB : il n'existe pas de "TR 9507" ; c'etait une citation erronee.
  - Love R.H. 1978, JASA 64(2):571-580 : resonance des vessies natatoires
  - Love R.H. 1971, JASA 49(3B):816-823 : target strength individuel
  - Francois & Garrison 1982, JASA 72(6):1879 : absorption en eau de mer
    (valable ~0.1 kHz - 1 MHz ; au-dela de ~200 kHz le terme MgSO4 peut etre
    sous-estime, le terme eau pure domine au-dela de ~500 kHz)
  - Chapman & Harris 1962, JASA 34(10):1592 : retrodiffusion de SURFACE
  - Klevjer et al. 2016, Sci. Rep. 6:19873 : profondeur jour de la DSL
    (moyenne ponderee 350-590 m selon les regions)
  - Ariza et al. 2016 (Canaries) : resonance de vessies observee a 18 kHz
    dans une DSL de migrantes a 400-500 m le jour
  - Modele simplifie type SEARAY : Sv = Sp + 7 log10(f), Sp in {-50,-70,-90} dB
  - Chlorophylle : EMODnet Chemistry DIVAnd Mediterranee (source "emodnet")
  - Chlorophylle : NASA MODIS Aqua L3 via ERDDAP CoastWatch (source "erddap")

Avertissement modelisation : la compression de la DSL sur le plateau
(dsl_layer) et les bornes jour/nuit (400/120 m) sont des parametrisations
de calibration, pas des equations publiees ; voir Klevjer et al. 2016
pour la variabilite reelle des profondeurs.

Usage: python reverberation_3d.py --chl emodnet --bathy auto
       python reverberation_3d.py --chl erddap --bathy auto
       python reverberation_3d.py --chl chl_golfe_lion_automne2021.csv
       python reverberation_3d.py --modele searay
       python reverberation_3d.py --freqs-log 0.1 1000 13 --bathy auto
       python reverberation_3d.py --freqs 0.1,12,1000 --hours 2,14
"""

import argparse
import copy
import datetime as dt
import json
import math
import os
import struct

import numpy as np
from scipy.io import netcdf_file

# ----------------------------------------------------------------------------
# CONFIGURATION (zone demo : Golfe du Lion, hiver)
# ----------------------------------------------------------------------------
CONFIG = {
    # Grille geographique
    "lat_min": 42.0, "lat_max": 43.2, "dlat": 0.05,     # deg
    "lon_min": 3.0,  "lon_max": 4.8,  "dlon": 0.05,      # deg
    # Grille verticale
    "depth_max": 1000.0, "ddepth": 10.0,                 # m
    # Frequences (kHz) : plage supportee 100 Hz (0.1 kHz) - 1 MHz (1000 kHz)
    "freqs_khz": [0.1, 0.5, 1.0, 2.0, 6.5, 12.0, 30.0, 100.0, 300.0, 1000.0],
    "freq_min_khz": 0.1,
    "freq_max_khz": 1000.0,
    # Date et heures UTC de la carte (diel de la couche de diffusion)
    "date": "2026-11-17",
    "hours_utc": list(range(0, 24)),
    # Environnement (Golfe du Lion, hiver)
    "temp_c": 13.0, "salinity_psu": 38.2, "ph": 8.0,
    # Modele DSL (APL-UW type)
    "z_dsl_day": 400.0,       # m, profondeur jour
    "z_dsl_night": 120.0,     # m, profondeur nuit (remontee)
    "dsl_sigma_m": 55.0,      # m, epaisseur (gaussienne)
    "dsl_peak_sv_db": -67.0,  # dB re 1 m^-1 au pic de resonance dans la couche
    "swimbladder_a0_mm": 2.5, # mm, rayon median des vessies
    "sigma_ln_a": 0.6,        # largeur lognormale des tailles
    "a_min_mm": 0.3, "a_max_mm": 30.0, "n_sizes": 64,  # a_max 30 mm : resonance vers 100 Hz en surface
    "q_factor": 4.0,          # facteur de qualite de resonance
    # Fond biologique (plancton / micronecton, spectre ~ plat)
    "bg_sv_surface_db": -100.0,  # dB re 1 m^-1 en surface
    "bg_scale_m": 250.0,          # decroissance exponentielle avec la profondeur
    # Champ de chlorophylle. Source UNIQUE (pas de melange) :
    #   "emodnet" : DIVAnd saisonnier Mediterranee (profils 0-600 m)
    #   "erddap"  : MODIS Aqua L3 via ERDDAP CoastWatch (surface satellite)
    #   chemin CSV : fichier deja telecharge, format reconnu
    #   None : proxy synthetique fonde sur la bathymetrie
    "chlorophyll": {
        "source": None,
        "emodnet_url": ("https://erddap.emodnet.eu/erddap/griddap/"
                        "oceanbrowser_opendap_3010_970d_4bf7.csv"),
        "erddap_url": ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
                       "erdMH1chlamday.csv"),
        "z_mix_m": 30.0,    # extension verticale (source surface seule) :
        "z_decay_m": 80.0,  # couche melee puis decroissance exponentielle
        "ref_mg_m3": 0.3,   # chl de reference : P = 1 (mg/m^3)
        "beta": 0.7,        # exposant de mise a l'echelle biomasse
        "p_min": 0.15, "p_max": 3.0,
    },
    # Bathymetrie : synthetique (demo), ou DTM EMODnet via WCS public (~115 m).
    # "bathymetry.source" : None | "auto" | chemin du GeoTIFF produit par auto.
    "bathy_shelf_m": 90.0, "bathy_deep_m": 1500.0, "bathy_slope_lon": 3.7,
    "bathymetry": {
        "source": None,
        "wcs": "https://ows.emodnet-bathymetry.eu/wcs",
        "coverage": "emodnet__mean",   # dernier DTM ; releases : _2016.._2022
        "buffer_deg": 0.05,
    },
    # Modele simplifie type SEARAY (Sp : -50 forte, -70 moyenne, -90 faible)
    "searay_sp_db": -70.0,
    # Exports GeoTIFF : tranches (frequence kHz, heures UTC, pas en m)
    "geotiff": {"freq_khz": 12.0, "hours_utc": [2, 14], "depth_step_m": 50.0},
    # Repertoire de sortie
    "outdir": "out",
}


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def deep_update(base, patch):
    """Fusion récursive de patch dans base (dicts imbriqués)."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def validate_freqs(freqs_khz, cfg):
    """Trie les frequences (kHz) et verifie la plage supportee 100 Hz - 1 MHz."""
    f = sorted(float(x) for x in freqs_khz)
    if f[0] < cfg["freq_min_khz"] - 1e-12 or f[-1] > cfg["freq_max_khz"] + 1e-12:
        raise ValueError("frequences hors plage supportee [%.3g, %.3g] kHz "
                         "(100 Hz - 1 MHz) : %s"
                         % (cfg["freq_min_khz"], cfg["freq_max_khz"], f))
    return f


def _urlretrieve(url, path, timeout=180):
    """Telechargement robuste : fichier temporaire puis renommage atomique."""
    import shutil
    import tempfile
    import urllib.request
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as r:
            shutil.copyfileobj(r, out)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


# ----------------------------------------------------------------------------
# Physique de base
# ----------------------------------------------------------------------------
def sound_speed(t_c, s_psu, z_m):
    """Celerite (m/s), formule type Mackenzie ( valide 0<T<35, 0<S<45 )."""
    return (1449.2 + 4.6 * t_c - 0.055 * t_c ** 2 + 0.00029 * t_c ** 3
            + (1.34 - 0.010 * t_c) * (s_psu - 35) + 0.016 * z_m)


def absorption_fg(f_khz, t_c, s_psu, zmax_m, ph=8.0):
    """Absorption Francois-Garrison 1982, dB/km.

    Valable ~0.1 kHz - 1 MHz. Au-dela de ~200 kHz le terme MgSO4 peut etre
    sous-estime ; le terme eau pure domine au-dela de ~500 kHz.
    """
    c = sound_speed(t_c, s_psu, 0.0)
    f2 = f_khz ** 2
    # acide borique
    f1 = 2.8 * math.sqrt(s_psu / 35.0) * 10 ** (4 - 1245.0 / (t_c + 273.15))
    a1 = (8.86 / c) * 10 ** (0.78 * ph - 5.0)
    alpha1 = a1 * 1.0 * f1 * f2 / (f1 ** 2 + f2)
    # sulfate de magnesium
    f2r = 8.17e8 * 10 ** (-1990.0 / (t_c + 273.15)) / (1 + 0.0018 * (s_psu - 35.0))
    a2 = 21.44 * s_psu / (c * (1 + 0.025 * t_c))
    p2 = 1 - 1.37e-4 * zmax_m + 6.2e-9 * zmax_m ** 2
    alpha2 = a2 * p2 * f2r * f2 / (f2r ** 2 + f2)
    # eau pure (viscosite)
    if t_c <= 20:
        a3 = (4.937e-4 - 2.59e-5 * t_c + 9.11e-7 * t_c ** 2 - 1.5e-8 * t_c ** 3)
    else:
        a3 = (3.964e-4 - 1.146e-5 * t_c + 1.45e-7 * t_c ** 2 - 6.5e-10 * t_c ** 3)
    p3 = 1 - 3.83e-5 * zmax_m + 4.9e-10 * zmax_m ** 2
    alpha3 = a3 * p3 * f2
    return alpha1 + alpha2 + alpha3


def rl_volume(sl_db, f_khz, r_m, tau_s, beamwidth_deg, sv_db, t_c, s_psu, ph=8.0):
    """Niveau de reverbération volumique (Urick), dB re 1 uPa.

    RL = SL - 2*TL + Sv + 10*log10(V),  V = (c*tau/2) * psi * r^2
    TL aller = 20*log10(r) + alpha*r/1000  (propagation spherique + FG)
    psi : angle solide equivalent du lobe (approx. petit angle, x*y en rad).
    """
    c = sound_speed(t_c, s_psu, r_m / 2.0)
    alpha = absorption_fg(f_khz, t_c, s_psu, r_m, ph) / 1000.0  # dB/m
    tl_one = 20.0 * math.log10(r_m) + alpha * r_m
    psi = (math.radians(beamwidth_deg[0]) * math.radians(beamwidth_deg[1]))
    vol = (c * tau_s / 2.0) * psi * r_m ** 2
    return sl_db - 2 * tl_one + sv_db + 10.0 * math.log10(vol)


# ----------------------------------------------------------------------------
# Position solaire (NOAA, formules simplifiees)
# ----------------------------------------------------------------------------
def solar_elevation(lat_deg, lon_deg, date, hour_utc):
    """Elevation du soleil (degres) ; positive le jour."""
    doy = date.timetuple().tm_yday
    gamma = 2 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma) - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    tst = (hour_utc * 60.0 + eqtime + 4.0 * lon_deg) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    lat_r = math.radians(lat_deg)
    return math.degrees(math.asin(
        math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)))


def dsl_depth(lat_deg, lon_deg, date, hour_utc, cfg):
    """Profondeur de la couche de diffusion : jour -> nuit (migration diel)."""
    elev = solar_elevation(lat_deg, lon_deg, date, hour_utc)
    # transition douce entre +2 deg (jour) et -12 deg (crepuscule nautique)
    s_up, s_dn = math.sin(math.radians(2.0)), math.sin(math.radians(-12.0))
    m = np.clip((s_up - math.sin(math.radians(elev))) / (s_up - s_dn), 0.0, 1.0)
    return cfg["z_dsl_day"] + m * (cfg["z_dsl_night"] - cfg["z_dsl_day"])


def dsl_layer(depth_m, z_nom, h_col, cfg):
    """Gaussienne de la DSL, comprimee si la colonne est trop peu profonde.

    Au large (h >> z_nom + 2 sigma) : couche nominale, epaisseur dsl_sigma_m.
    Sur le plateau (h < z_nom) : les organismes ne peuvent pas descendre
    plus bas que le fond ; la couche se comprime dans la colonne disponible :
      - centre ramene a h - clear (clear = 15 % de la colonne, >= 10 m),
      - epaisseur limitee a h/3,
      - amplitude renormalisee a biomasse integree constante
        (gain = sigma_nominal / sigma_effectif, Sv volumique plus eleve).

    depth_m : (nz,) ; z_nom, h_col : scalaires ou tableaux broadcastables.
    Retourne la couche normalisee, meme broadcast que z_nom/h_col.
    """
    sig_nom = float(cfg["dsl_sigma_m"])
    sig = np.maximum(np.minimum(sig_nom, h_col / 3.0), 5.0)
    clear = np.maximum(0.15 * h_col, 10.0)
    z_eff = np.maximum(np.minimum(z_nom, h_col - clear), 0.5 * sig)
    return np.exp(-0.5 * ((depth_m - z_eff) / sig) ** 2) * (sig_nom / sig)


# ----------------------------------------------------------------------------
# Bathymetrie (synthetique pour la demo)
# ----------------------------------------------------------------------------
def bathymetry(lat, lon, cfg):
    """Fond (m) sur la grille. Remplacer par EMODnet/GEBCO pour la vraie zone."""
    x = (lon - cfg["bathy_slope_lon"]) / 0.55
    shelf = 1.0 / (1.0 + np.exp(-x))          # talus oriente est
    h = cfg["bathy_shelf_m"] + (cfg["bathy_deep_m"] - cfg["bathy_shelf_m"]) * shelf
    ondul = 1.0 - 0.06 * np.cos((lat - 42.6) / 0.6 * math.pi)  # legere ondulation
    return h[None, :] * ondul[:, None]                            # (nlat, nlon)


def productivity(h_bathy):
    """Proxy de productivite (eaux neritiques > eaux du large), sans dimension."""
    return 1.0 + 0.9 * np.exp(-h_bathy / 250.0)


# ----------------------------------------------------------------------------
# Champ de chlorophylle : source EMODnet (DIVAnd Med) OU ERDDAP (MODIS Aqua)
# ----------------------------------------------------------------------------
def season_tag(date):
    """Saison EMODnet couvrant la date (pas de temps au 16 fev/mai/aout/nov)."""
    m = date.month
    if m in (12, 1, 2):
        return "02-16"
    if m in (3, 4, 5):
        return "05-16"
    if m in (6, 7, 8):
        return "08-16"
    return "11-16"


def download_chlorophyll_emodnet(path, cfg, date, lat, lon):
    """Source EMODnet : CSV ERDDAP pour la zone et la saison de la date.

    DIVAnd saisonnier Mediterranee (profils 0-600 m), saisons 1992 a 2021.
    """
    from urllib.parse import quote
    s = season_tag(date)
    year = min(date.year, 2021)  # jeu de donnees : saisons 1992 a 2021
    query = ("?Water_body_chlorophyll_a"
             "[(%d-%s):(%d-%s)][(0):(600)][(%.3f):(%.3f)][(%.3f):(%.3f)]"
             % (year, s, year, s,
                float(lat.min()) - 0.25, float(lat.max()) + 0.25,
                float(lon.min()) - 0.25, float(lon.max()) + 0.25))
    url = cfg["chlorophyll"]["emodnet_url"] + quote(query, safe="?()[]:.,-=&")
    print("Telechargement chlorophylle EMODnet :", url)
    _urlretrieve(url, path)
    return path


# Alias historique : --chl auto designait la source EMODnet.
download_chlorophyll = download_chlorophyll_emodnet


def download_chlorophyll_erddap(path, cfg, date, lat, lon):
    """Source ERDDAP CoastWatch : MODIS Aqua L3 mondial 4 km.

    Climatologie du mois de la date : produit journalier echantillonne au
    pas 12 (tous les memes jours de mois 2003-2021), chlorophylle de
    surface satellite uniquement.
    """
    from urllib.parse import quote
    mm = "%02d" % date.month
    query = ("?chlorophyll"
             "[(2003-%s-16T00:00:00Z):12:(last)]"
             "[(%.3f):(%.3f)][(%.3f):(%.3f)]"
             % (mm,
                float(lat.min()) - 0.25, float(lat.max()) + 0.25,
                float(lon.min()) - 0.25, float(lon.max()) + 0.25))
    url = cfg["chlorophyll"]["erddap_url"] + quote(query, safe="?()[]:.,-=&")
    print("Telechargement chlorophylle ERDDAP (MODIS Aqua) :", url)
    _urlretrieve(url, path)
    return path


def read_chlorophyll_modis_csv(path):
    """CSV ERDDAP MODIS -> (lats, lons, chl2d[nlat, nlon] mg/m^3).

    Climatologie : moyenne sur tous les pas de temps du fichier.
    Les pixels terrestres sont absents du CSV (pas de ligne), la grille
    construite ne couvre donc que les pixels océaniques valides ; les
    points du modele hors de cette enveloppe sont extrapoles au plus
    proche voisin par l'interpolateur.
    """
    lines = open(path).read().splitlines()
    hdr = lines[0].split(",")
    ila, ilo, iv = (hdr.index("latitude"), hdr.index("longitude"),
                    hdr.index("chlorophyll"))
    acc = {}
    for ln in lines[2:]:
        if not ln:
            continue
        r = ln.split(",")
        key = (float(r[ila]), float(r[ilo]))
        if key not in acc:
            acc[key] = []
        v = r[iv]
        if v in ("", "NaN"):
            continue
        fv = float(v)
        if not (fv > 0.0):          # artefacts negatifs du capteur
            continue
        acc[key].append(fv)
    if not any(acc.values()):
        raise ValueError("Aucune valeur de chlorophylle dans " + path)
    lats = sorted(set(k[0] for k in acc))
    lons = sorted(set(k[1] for k in acc))
    ai = {a: k for k, a in enumerate(lats)}
    oi = {o: k for k, o in enumerate(lons)}
    chl = np.full((len(lats), len(lons)), np.nan)
    for (a, o), vs in acc.items():
        if vs:
            chl[ai[a], oi[o]] = float(np.mean(vs))
    return np.array(lats), np.array(lons), chl


def chlorophyll_grid_surface(lats, lons, chl2d, depth_m, lat, lon, cfg):
    """Chlorophylle de surface (satellite) -> champ 3D (nz, nlat, nlon).

    Extension verticale : valeur de surface maintenue dans la couche melee
    (z <= z_mix_m), puis decroissance exponentielle d'echelle z_decay_m.
    Coordonnees bornees a la grille source (extrapolation constante hors
    enveloppe), valeurs negatives mises a 0.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    c = chl2d.copy()
    if np.isnan(c).any():
        ind = distance_transform_edt(np.isnan(c), return_distances=False,
                                     return_indices=True)
        c = c[tuple(ind)]
    interp = RegularGridInterpolator((lats, lons), c,
                                     bounds_error=False, fill_value=None)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    la = np.clip(la, float(lats[0]), float(lats[-1]))
    lo = np.clip(lo, float(lons[0]), float(lons[-1]))
    surf = np.clip(interp(np.stack([la, lo], -1)), 0.0, None)  # (nlat, nlon)
    cc = cfg["chlorophyll"]
    f_z = np.where(depth_m <= cc["z_mix_m"], 1.0,
                   np.exp(-(depth_m - cc["z_mix_m"]) / cc["z_decay_m"]))
    return surf[None, :, :] * f_z[:, None, None]


def read_chlorophyll_csv(path):
    """CSV ERDDAP -> (depths, lats, lons, chl[nz, nlat, nlon] mg/m^3).

    Si le CSV contient plusieurs pas de temps (saisons), le dernier est garde.
    """
    lines = open(path).read().splitlines()
    hdr = lines[0].split(",")
    it = hdr.index("time")
    id_, ila, ilo, iv = (hdr.index("depth"), hdr.index("latitude"),
                         hdr.index("longitude"), hdr.index("Water_body_chlorophyll_a"))
    times = sorted(set(ln.split(",")[it] for ln in lines[2:] if ln))
    t_last = times[-1]
    rows = [ln.split(",") for ln in lines[2:] if ln and ln.split(",")[it] == t_last]
    depths = sorted(set(float(r[id_]) for r in rows))
    lats = sorted(set(float(r[ila]) for r in rows))
    lons = sorted(set(float(r[ilo]) for r in rows))
    di = {d: k for k, d in enumerate(depths)}
    ai = {a: k for k, a in enumerate(lats)}
    oi = {o: k for k, o in enumerate(lons)}
    chl = np.full((len(depths), len(lats), len(lons)), np.nan)
    for r in rows:
        v = float(r[iv])
        if abs(v - 9.96921e36) < 1e30:
            v = np.nan
        chl[di[float(r[id_])], ai[float(r[ila])], oi[float(r[ilo])]] = v
    return np.array(depths), np.array(lats), np.array(lons), chl


def chlorophyll_grid(depths, lats, lons, chl, depth_m, lat, lon):
    """Chlorophylle interpolee sur la grille du modele (nz, nlat, nlon).

    Interpolation lineaire ; trous (terre) remplis par plus proche voisin ;
    coordonnees bornees a la grille source : sous 600 m (ou hors enveloppe)
    la valeur du niveau/coin le plus proche est prolongee (extrapolation
    constante) ; resultat borne a >= 0 (evite les NaN de (chl<0)**beta).
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    c = chl.copy()
    if np.isnan(c).any():
        ind = distance_transform_edt(np.isnan(c), return_distances=False,
                                     return_indices=True)
        c = c[tuple(ind)]
    interp = RegularGridInterpolator((depths, lats, lons), c,
                                     bounds_error=False, fill_value=None)
    z, la, lo = np.meshgrid(depth_m, lat, lon, indexing="ij")
    z = np.clip(z, float(depths[0]), float(depths[-1]))
    la = np.clip(la, float(lats[0]), float(lats[-1]))
    lo = np.clip(lo, float(lons[0]), float(lons[-1]))
    out = interp(np.stack([z, la, lo], -1))
    return np.clip(out, 0.0, None)


def productivity_from_chl(chl_grid, depth_m, cfg, dsl_range=(0.0, 500.0)):
    """Facteur de productivite P a partir de la chlorophylle.

    P = clip((chl/ref)^beta, p_min, p_max), P = 1 pour chl = ref_mg_m3.
    chl est borne a >= 0 avant la puissance (beta non entier).
    Retourne (p3(nz,nlat,nlon), p_col(nlat,nlon)) : fond biologique et DSL
    (moyenne de P sur la bande de migration de la couche de diffusion).
    """
    cc = cfg["chlorophyll"]
    c = np.clip(chl_grid, 0.0, None)
    p3 = np.clip((c / cc["ref_mg_m3"]) ** cc["beta"],
                 cc["p_min"], cc["p_max"])              # (nz, nlat, nlon)
    m = (depth_m >= dsl_range[0]) & (depth_m <= dsl_range[1])
    p_col = p3[m].mean(axis=0)                          # (nlat, nlon)
    return p3, p_col


# ----------------------------------------------------------------------------
# Bathymetrie EMODnet (WCS 2.0.1 public, GeoTIFF maille 1/16' ~ 115 m)
# ----------------------------------------------------------------------------
def _snap(v, res=1.0 / 960.0):
    """Aligne une coordonnee sur la maille native du DTM (1/16' d'arc)."""
    return round(round(v / res) * res, 6)


def download_bathymetry(outdir, cfg, lat, lon):
    """Telecharge le DTM EMODnet pour la grille + buffer, retourne le chemin.

    Le nom de fichier embarque l'etendue (bathy_latmin_latmax_lonmin_lonmax)
    car le GeoTIFF WCS n'inclut pas de geotransform.
    """
    import os
    buf = cfg["bathymetry"]["buffer_deg"]
    la0, la1 = _snap(float(lat.min()) - buf), _snap(float(lat.max()) + buf)
    lo0, lo1 = _snap(float(lon.min()) - buf), _snap(float(lon.max()) + buf)
    q = ("?service=WCS&version=2.0.1&request=GetCoverage&coverageId=%s"
         "&SUBSET=Lat(%.6f,%.6f)&SUBSET=Long(%.6f,%.6f)&FORMAT=image%%2Ftiff"
         % (cfg["bathymetry"]["coverage"], la0, la1, lo0, lo1))
    url = cfg["bathymetry"]["wcs"] + q
    tif = os.path.join(outdir, "bathy_%.6f_%.6f_%.6f_%.6f.tif" % (la0, la1, lo0, lo1))
    if os.path.exists(tif):
        print("Bathymetrie EMODnet deja telechargee :", tif)
        return tif
    print("Telechargement bathymetrie EMODnet :", url)
    _urlretrieve(url, tif)
    return tif


def read_bathymetry_tif(path):
    """GeoTIFF EMODnet -> (lats croissant, lons croissant, profondeur en m).

    L'etendue est deduite du nom bathy_<latmin>_<latmax>_<lonmin>_<lonmax>.tif.
    Le WCS fournit des profondeurs negatives en mer (positives sur terre),
    ligne 0 = nord, sans geotransform.
    """
    import re
    from PIL import Image
    m = re.match(r".*bathy_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)\.tif$",
                 path)
    if not m:
        raise ValueError("bbox illisible : %s (utiliser --bathy auto d'abord)" % path)
    la0, la1, lo0, lo1 = (float(m.group(i)) for i in range(1, 5))
    a = np.array(Image.open(path)).astype(float)     # ligne 0 = nord
    h, w = a.shape
    lats = (la1 - (np.arange(h) + 0.5) * (la1 - la0) / h)[::-1]
    lons = lo0 + (np.arange(w) + 0.5) * (lo1 - lo0) / w
    depth = np.clip(-a[::-1], 0.0, None)              # lat croissantes, m >= 0
    return lats, lons, depth


def bathy_on_grid(lats_b, lons_b, depth_b, lat, lon):
    """Interpole le DTM sur la grille (nlat, nlon) ; trous au plus proche."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    c = depth_b.copy()
    if np.isnan(c).any():
        ind = distance_transform_edt(np.isnan(c), return_distances=False,
                                      return_indices=True)
        c = c[tuple(ind)]
    interp = RegularGridInterpolator((lats_b, lons_b), c, bounds_error=False,
                                     fill_value=None)
    la2, lo2 = np.meshgrid(lat, lon, indexing="ij")
    return np.clip(interp(np.stack([la2, lo2], -1)), 0.0, None)


# ----------------------------------------------------------------------------
# Modele de retrodiffusion volumique S_v
# ----------------------------------------------------------------------------
def bladder_spectrum(freqs_khz, depth_m, cfg):
    """W(f, z) : section efficace retrodiffusante ponderee par les tailles (m^2).

    Resonance de vessie : f_r(a,z) = 1/(2*pi*a) * sqrt(3*gamma*P(z)/rho)
    Reponse : passe-bande du 2nd ordre, pic k*a^2, facteur de qualite Q.
    Couvre 100 Hz (grandes vessies, ~30 mm en surface) a 1 MHz (petites
    vessies profondes) ; hors resonance la reponse decroit en f^2 (basse
    frequence) ou 1/f^2 (haute frequence).
    """
    freqs_khz = np.asarray(freqs_khz, float)
    rho, gamma, p0, g = 1027.0, 1.4, 101325.0, 9.81
    a = np.logspace(math.log10(cfg["a_min_mm"] * 1e-3),
                    math.log10(cfg["a_max_mm"] * 1e-3), cfg["n_sizes"])
    p = np.exp(-0.5 * (np.log(a / (cfg["swimbladder_a0_mm"] * 1e-3))
                       / cfg["sigma_ln_a"]) ** 2)
    p /= p.sum()
    f = freqs_khz[:, None, None] * 1e3                       # (nf,1,1)
    a3 = a[None, None, :]                                    # (1,1,na)
    z3 = depth_m[None, :, None]                              # (1,nz,1)
    press = p0 + rho * g * z3
    f_r = (1.0 / (2 * math.pi * a3)) * np.sqrt(3 * gamma * press / rho)
    q = cfg["q_factor"]
    x = (f ** 2 - f_r ** 2) / (f * f_r / q)
    sigma = a3 ** 2 / (1 + x ** 2)                            # pic normalise a^2
    return np.einsum("fza,a->fz", sigma, p)                    # (nf, nz)


def sv_field(freqs_khz, depth_m, lat, lon, date, hours_utc, cfg, h_bathy,
             p3=None, p_col=None):
    """S_v 3D en lineaire (m^-1) : DSL (resonance + diel) + fond biologique.

    Retourne (nt, nf, nz, nlat, nlon) en dB re 1 m^-1 (NaN sous le fond).
    Sur le plateau, la DSL est comprimee dans la colonne disponible
    (voir dsl_layer) : biomasse conservee, Sv volumique plus eleve.
    Si p3/p_col sont fournis (chlorophylle), ils donnent la structure spatiale
    et verticale ; sinon on retombe sur le proxy synthetique de bathymetrie.
    """
    nf, nz = len(freqs_khz), len(depth_m)
    nlat, nlon = lat.size, lon.size
    # spectre de resonance par profondeur (ne depend pas de lat/lon)
    w = bladder_spectrum(np.asarray(freqs_khz, float), depth_m, cfg)   # (nf, nz)
    # calibration : pic de -dsl_peak_sv_db a la resonance au centre de la couche jour
    z_c = cfg["z_dsl_day"]
    a0 = cfg["swimbladder_a0_mm"] * 1e-3
    rho, gamma, p0, g = 1027.0, 1.4, 101325.0, 9.81
    f_res = (1 / (2 * math.pi * a0)) * math.sqrt(3 * gamma * (p0 + rho * g * z_c) / rho)
    w_ref = np.interp(f_res / 1e3, freqs_khz, w[:, np.argmin(np.abs(depth_m - z_c))])
    n0 = 10 ** (cfg["dsl_peak_sv_db"] / 10.0) / max(w_ref, 1e-30)
    # fond biologique : decay exponentiel (proxy) ou profil de chlorophylle
    bg0 = 10 ** (cfg["bg_sv_surface_db"] / 10.0)
    if p3 is not None:
        bgz = bg0 * p3.transpose(1, 2, 0)               # (nlat, nlon, nz)
    else:
        bgz = bg0 * np.exp(-depth_m / cfg["bg_scale_m"])  # (nz,)
    sv = np.empty((len(hours_utc), nf, nz, nlat, nlon), np.float32)
    for it, hu in enumerate(hours_utc):
        z_dsl = np.empty((nlat, nlon))
        for i in range(nlat):
            for j in range(nlon):
                z_dsl[i, j] = dsl_depth(float(lat[i]), float(lon[j]), date, hu, cfg)
        # couche de diffusion, comprimee sur le plateau (colonne trop courte)
        layer = dsl_layer(depth_m[None, None, :], z_dsl[:, :, None],
                          h_bathy[:, :, None], cfg)          # (nlat,nlon,nz)
        # Sv lineaire (m^-1) : DSL resonance + fond biologique
        a_term = layer[..., None] * (n0 * w).T[None, None, :, :]      # (nlat,nlon,nz,nf)
        if p3 is not None:
            lin = (p_col[:, :, None, None] * a_term
                   + bgz[:, :, :, None])                    # (nlat,nlon,nz,nf)
        else:
            prod = productivity(h_bathy)                     # (nlat, nlon)
            lin = prod[:, :, None, None] * (a_term + bgz[None, None, :, None])
        sv[it] = 10.0 * np.log10(lin).transpose(3, 2, 0, 1)           # (nf,nz,nlat,nlon)
    return sv


# ----------------------------------------------------------------------------
# Ecriture netCDF (classic/64-bit via scipy)
# ----------------------------------------------------------------------------
def write_netcdf(path, sv_db, lat, lon, depth_m, freqs_khz, date, hours_utc, cfg,
                 chl_grid=None, h_bathy=None, chl_source=None):
    nt, nf, nz, nlat, nlon = sv_db.shape
    with netcdf_file(path, "w", version=2) as f:
        f.title = ("Reverberation volumique Sv 3D (lat/lon/immersion), "
                   "modele type APL-UW : resonance de vessies + migration diel")
        f.Conventions = "CF-1.6"
        f.institution = "SEAGNAL"
        f.history = f"{date.isoformat()} genere par reverberation_3d.py"
        f.references = ("Urick 1983; APL-UW TR 9407 (Jackson et al. 1994); "
                        "Love 1971/1978; Francois & Garrison 1982")
        f.comment = ("S_v en dB re 1 m^-1 (retrodiffusion volumique). "
                     "NaN = sous le fond bathymetrique.")
        f.createDimension("time", None)
        f.createDimension("frequency", nf)
        f.createDimension("depth", nz)
        f.createDimension("lat", nlat)
        f.createDimension("lon", nlon)
        v = f.createVariable("time", "f", ("time",))
        v.units = f"hours since {date.isoformat()} 00:00:00 UTC"
        v.standard_name = "time"
        v[:] = np.asarray(hours_utc, "f4")
        v = f.createVariable("frequency", "f", ("frequency",))
        v.units = "kHz"
        v[:] = np.asarray(freqs_khz, "f4")
        v = f.createVariable("depth", "f", ("depth",))
        v.units = "m"
        v.long_name = "immersion (positive vers le bas)"
        v[:] = np.asarray(depth_m, "f4")
        v = f.createVariable("lat", "f", ("lat",))
        v.units = "degrees_north"
        v[:] = np.asarray(lat, "f4")
        v = f.createVariable("lon", "f", ("lon",))
        v.units = "degrees_east"
        v[:] = np.asarray(lon, "f4")
        v = f.createVariable("Sv", "f",
                             ("time", "frequency", "depth", "lat", "lon"))
        v.units = "dB re 1 m^-1"
        v.long_name = "volume_backscattering_strength"
        v.coordinates = "time frequency depth lat lon"
        v[:] = sv_db.astype("f4")
        if chl_grid is not None:
            v = f.createVariable("chlorophyll", "f", ("depth", "lat", "lon"))
            v.units = "mg m-3"
            v.long_name = {
                "emodnet": "chlorophyll-a (EMODnet DIVAnd saisonnier)",
                "erddap": ("chlorophyll-a (MODIS Aqua L3, "
                           "extension verticale expo)"),
            }.get(chl_source, "chlorophyll-a")
            v[:] = chl_grid.astype("f4")
            f.chlorophyll_source = chl_source or "inconnu"
        if h_bathy is not None:
            v = f.createVariable("bathymetry", "f", ("lat", "lon"))
            v.units = "m"
            v.long_name = "profondeur du fond (DTM EMODnet ou demo)"
            v[:] = h_bathy.astype("f4")
    return path


# ----------------------------------------------------------------------------
# Ecriture GeoTIFF (writer minimal, WGS84, float32)
# ----------------------------------------------------------------------------
def write_geotiff(path, grid, lat, lon):
    """grid : (nlat, nlon) float32, ligne 0 = lat max (nord en haut).

    Les tableaux du modele ont la ligne 0 au SUD (lat croissante) : l'appelant
    doit passer grid[::-1].
    """
    h, w = grid.shape
    data = np.ascontiguousarray(grid, dtype="<f4").tobytes()
    dx = float(lon[1] - lon[0])
    dy = float(lat[1] - lat[0])
    lon0, lat_top = float(lon[0]), float(lat[-1])
    geokeys = [
        1, 1, 0, 3,          # version 1.1.0, 3 cles
        1024, 0, 1, 2,       # GTModelTypeGeoKey = ModelTypeGeographic
        1025, 0, 1, 1,       # GTRasterTypeGeoKey = RasterPixelIsArea
        2048, 0, 1, 4326,    # GeographicTypeGeoKey = WGS84
    ]
    pixel_scale = struct.pack("<3d", dx, dy, 0.0)
    tiepoint = struct.pack("<6d", 0.0, 0.0, 0.0, lon0, lat_top, 0.0)
    ascii_params = b"nan\x00"  # nodata GDAL

    entries = []  # (tag, type, count, value_bytes)

    def ent(tag, typ, count, payload):
        entries.append([tag, typ, count, payload])

    ent(256, 4, 1, struct.pack("<I", w))           # ImageWidth
    ent(257, 4, 1, struct.pack("<I", h))           # ImageLength
    ent(258, 3, 1, struct.pack("<H", 32))          # BitsPerSample
    ent(259, 3, 1, struct.pack("<H", 1))           # Compression = none
    ent(262, 3, 1, struct.pack("<H", 1))           # Photometric = min-is-black
    ent(273, 4, 1, None)                           # StripOffsets (rempli apres)
    ent(277, 3, 1, struct.pack("<H", 1))           # SamplesPerPixel
    ent(278, 4, 1, struct.pack("<I", h))           # RowsPerStrip
    ent(279, 4, 1, struct.pack("<I", w * h * 4))   # StripByteCounts
    ent(284, 3, 1, struct.pack("<H", 1))           # PlanarConfig = chunky
    ent(33550, 12, 3, pixel_scale)                  # ModelPixelScaleTag
    ent(33922, 12, 6, tiepoint)                    # ModelTiepointTag
    ent(339, 3, 1, struct.pack("<H", 3))            # SampleFormat = IEEE float
    ent(34735, 3, len(geokeys), struct.pack("<%dH" % len(geokeys), *geokeys))
    ent(42113, 2, 4, ascii_params)                  # GDAL_NODATA

    # layout : header(8) + IFD + valeurs externes + pixels
    ifd_off = 8
    ifd_len = 2 + 12 * len(entries) + 4
    ext_off = ifd_off + ifd_len
    off = ext_off
    payloads = {}
    for e in entries:
        tag, typ, count, payload = e
        if tag == 273:
            continue  # strip offsets plus tard
        if len(payload) > 4:
            payloads[tag] = off
            off += len(payload) + (4 - len(payload) % 4) % 4
    strip_off = off
    for e in entries:
        if e[0] == 273:
            e[3] = struct.pack("<I", strip_off)

    out = bytearray(b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd_off))
    out += struct.pack("<H", len(entries))
    for tag, typ, count, payload in entries:
        if payload is None:
            raise RuntimeError("tag non rempli: %d" % tag)
        if len(payload) <= 4:
            value = payload + b"\x00" * (4 - len(payload))
            out += struct.pack("<HHI", tag, typ, count) + value
        else:
            out += struct.pack("<HHI", tag, typ, count) + struct.pack("<I", payloads[tag])
    out += struct.pack("<I", 0)  # pas d'IFD suivant
    for e in entries:
        if e[0] in payloads:
            out += e[3]
            pad = (4 - len(e[3]) % 4) % 4
            out += b"\x00" * pad
    out += data
    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


# ----------------------------------------------------------------------------
# Quicklook PNG
# ----------------------------------------------------------------------------
def quicklook_png(path, sv_db, depth_m, freqs_khz, lat, lon, hours_utc, cfg,
                  chl_grid=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it_night = hours_utc.index(2) if 2 in hours_utc else 0
    it_day = hours_utc.index(14) if 14 in hours_utc else len(hours_utc) // 2
    i, j = len(lat) // 2, len(lon) // 2   # point central
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for k, (it, lab) in enumerate([(it_night, "nuit (02h UTC)"), (it_day, "jour (14h UTC)")]):
        for ifr, f in enumerate(freqs_khz):
            ax[0, k].plot(sv_db[it, ifr, :, i, j], -depth_m, label="%.4g kHz" % f)
        ax[0, k].set_title("Profil S_v au point central, %s" % lab)
        ax[0, k].set_xlabel("S_v (dB re 1 m^-1)")
        ax[0, k].set_ylabel("immersion (m)")
        ax[0, k].legend(fontsize=8)
        ax[0, k].grid(alpha=0.3)
    ifr12 = min(range(len(freqs_khz)), key=lambda q: abs(freqs_khz[q] - 12.0))
    zc = int(np.argmin(np.abs(depth_m - cfg["z_dsl_night"])))
    m = ax[1, 0].imshow(sv_db[it_night, ifr12, zc], origin="lower",
                       extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto",
                       cmap="viridis")
    ax[1, 0].set_title("Carte a %d m, %g kHz, nuit" % (depth_m[zc], freqs_khz[ifr12]))
    ax[1, 0].set_ylabel("lat (deg N)")
    plt.colorbar(m, ax=ax[1, 0], label="S_v (dB re 1 m^-1)")
    if chl_grid is not None:
        mc = ax[1, 1].imshow(np.log10(np.clip(chl_grid[0], 1e-6, None)), origin="lower",
                             extent=[lon[0], lon[-1], lat[0], lat[-1]],
                             aspect="auto", cmap="viridis")
        ax[1, 1].set_title("Chlorophylle-a de surface (mg/m^3, log10)")
        ax[1, 1].set_ylabel("lat (deg N)")
        plt.colorbar(mc, ax=ax[1, 1], label="log10(chl)")
    else:
        ax[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(cfg, modele="apl_uw", chl_src=None, bathy_src=None,
         freqs=None, hours=None, date_str=None):
    date = dt.datetime.strptime(date_str or cfg["date"], "%Y-%m-%d")
    lat = np.arange(cfg["lat_min"], cfg["lat_max"] + 1e-9, cfg["dlat"])
    lon = np.arange(cfg["lon_min"], cfg["lon_max"] + 1e-9, cfg["dlon"])
    depth = np.arange(0.0, cfg["depth_max"] + 1e-9, cfg["ddepth"])
    freqs = validate_freqs(freqs or cfg["freqs_khz"], cfg)
    hours = list(hours) if hours else list(cfg["hours_utc"])
    n_elem = len(hours) * len(freqs) * len(depth) * lat.size * lon.size
    if n_elem > 2e8:
        raise ValueError("grille trop grosse (%d elements, float32 ~%.0f Mo) : "
                         "reduire heures/frequences/profondeur/mailles"
                         % (n_elem, 4 * n_elem / 1e6))
    print("Grille : %d lat x %d lon x %d profondeur x %d frequences (%.3g-%.3g kHz) x %d heures"
          % (lat.size, lon.size, len(depth), len(freqs), freqs[0], freqs[-1], len(hours)))
    outdir = cfg["outdir"]
    os.makedirs(outdir, exist_ok=True)
    gt_dir = os.path.join(outdir, "geotiff")
    os.makedirs(gt_dir, exist_ok=True)

    # bathymetrie : DTM EMODnet (WCS) ou synthetique
    bathy_src = bathy_src or cfg["bathymetry"]["source"]
    if bathy_src:
        if bathy_src == "auto":
            tif = download_bathymetry(outdir, cfg, lat, lon)
        else:
            tif = bathy_src
        lats_b, lons_b, depth_b = read_bathymetry_tif(tif)
        h_bathy = bathy_on_grid(lats_b, lons_b, depth_b, lat, lon)
        print("Bathymetrie EMODnet interpolee : %.0f m a %.0f m"
              % (h_bathy.min(), h_bathy.max()))
    else:
        h_bathy = bathymetry(lat, lon, cfg)              # (nlat, nlon)
        print("Bathymetrie demo synthetique : %.0f m a %.0f m"
              % (h_bathy.min(), h_bathy.max()))

    # champ de chlorophylle (optionnel). Source unique, pas de melange :
    # "emodnet" (DIVAnd Med) ou "erddap" (MODIS Aqua mondial) ou un CSV.
    chl_grid, p3, p_col, chl_source = None, None, None, None
    if chl_src and chl_src != "none":
        if chl_src in ("emodnet", "auto"):     # auto = alias historique
            chl_csv = os.path.join(outdir, "chlorophyll_emodnet.csv")
            if not os.path.exists(chl_csv):
                download_chlorophyll_emodnet(chl_csv, cfg, date, lat, lon)
            chl_source = "emodnet"
        elif chl_src == "erddap":
            chl_csv = os.path.join(outdir, "chlorophyll_erddap_modis.csv")
            if not os.path.exists(chl_csv):
                download_chlorophyll_erddap(chl_csv, cfg, date, lat, lon)
            chl_source = "erddap"
        else:
            chl_csv = chl_src
            hdr = open(chl_csv).readline()
            chl_source = ("emodnet" if "Water_body_chlorophyll_a" in hdr
                          else "erddap")
        if chl_source == "emodnet":
            d_c, la_c, lo_c, chl_raw = read_chlorophyll_csv(chl_csv)
            chl_grid = chlorophyll_grid(d_c, la_c, lo_c, chl_raw,
                                        depth, lat, lon)
        else:
            la_c, lo_c, chl_raw = read_chlorophyll_modis_csv(chl_csv)
            chl_grid = chlorophyll_grid_surface(la_c, lo_c, chl_raw,
                                                depth, lat, lon, cfg)
        p3, p_col = productivity_from_chl(chl_grid, depth, cfg)
        print("Chlorophylle interpolee (source %s) : %.3f a %.3f mg/m^3"
              % (chl_source, float(np.nanmin(chl_grid)), float(np.nanmax(chl_grid))))

    if modele == "searay":
        # modele simplifie : Sv = Sp + 7 log10(f), constant sur la colonne
        nt, nf, nz, nlat, nlon = len(hours), len(freqs), len(depth), lat.size, lon.size
        sv = np.empty((nt, nf, nz, nlat, nlon), np.float32)
        base = cfg["searay_sp_db"] + 7.0 * np.log10(np.array(freqs))
        for iz, z in enumerate(depth):
            mask = z <= h_bathy                                  # (nlat, nlon)
            val = np.where(mask, 1.0, np.nan)
            sv[:, :, iz] = base[None, :, None, None] * val[None, None, :, :]
        sv_db = sv
    else:
        sv_db = sv_field(freqs, depth, lat, lon, date, hours, cfg, h_bathy,
                         p3=p3, p_col=p_col)
    print("S_v : %.1f a %.1f dB re 1 m^-1 (percentiles 1-99, hors NaN)"
          % tuple(np.nanpercentile(sv_db, [1, 99])))

    # masque sous le fond pour le modele APL-UW aussi
    below = depth[None, None, :, None, None] > h_bathy[None, None, None, :, :]
    sv_db = np.where(below, np.nan, sv_db).astype(np.float32)

    nc_path = os.path.join(outdir, "reverb_volume_3d.nc")
    write_netcdf(nc_path, sv_db, lat, lon, depth, freqs, date, hours, cfg,
                 chl_grid=chl_grid, h_bathy=h_bathy, chl_source=chl_source)
    print("netCDF ecrit :", nc_path, "%.1f Mo" % (os.path.getsize(nc_path) / 1e6))

    # exports GeoTIFF (ligne 0 = nord : on retourne les tranches, dont la
    # ligne 0 est au sud dans sv_db)
    g = cfg["geotiff"]
    ifr = min(range(len(freqs)), key=lambda q: abs(freqs[q] - g["freq_khz"]))
    zs = [z for z in depth if abs(z % g["depth_step_m"]) < 1e-9]
    n_tif = 0
    for hu in g["hours_utc"]:
        it = min(range(len(hours)), key=lambda q: abs(hours[q] - hu))
        for z in zs:
            iz = int(np.argmin(np.abs(depth - z)))
            sl = sv_db[it, ifr, iz]                      # (nlat, nlon), ligne 0 = sud
            if np.all(np.isnan(sl)):
                continue
            name = "Sv_f%gkHz_z%04dm_%02dhUTC.tif" % (freqs[ifr], z, hours[it])
            write_geotiff(os.path.join(gt_dir, name), sl[::-1], lat, lon)
            n_tif += 1
    print("GeoTIFF ecrits :", n_tif, "dans", gt_dir)

    png = quicklook_png(os.path.join(outdir, "quicklook.png"), sv_db, depth,
                        freqs, lat, lon, hours, cfg, chl_grid=chl_grid)
    print("Quicklook :", png)

    # utilitaire RL (exemples repartis sur 100 Hz - 1 MHz)
    examples = []
    for f, r, tau, bw in [(0.1, 20000.0, 0.5, (8.0, 8.0)),
                          (6.5, 8000.0, 0.05, (12.0, 12.0)),
                          (12.0, 4000.0, 0.05, (10.0, 10.0)),
                          (100.0, 800.0, 0.01, (5.0, 5.0)),
                          (1000.0, 150.0, 0.002, (2.0, 2.0))]:
        it = hours.index(14) if 14 in hours else hours[0]
        ifr = min(range(len(freqs)), key=lambda q: abs(freqs[q] - f))
        zsl = depth[np.nanargmax(sv_db[it, ifr, :, len(lat) // 2, len(lon) // 2])]
        sv_mid = float(sv_db[it, ifr, int(np.argmin(np.abs(depth - zsl))),
                         len(lat) // 2, len(lon) // 2])
        rl = rl_volume(220.0, f, r, tau, bw, sv_mid, cfg["temp_c"], cfg["salinity_psu"])
        alpha = absorption_fg(f, cfg["temp_c"], cfg["salinity_psu"], r, cfg["ph"])
        examples.append({"f_khz": f, "r_m": r, "Sv_dB": sv_mid, "alpha_dB_km": alpha,
                         "RL_dB": rl})
        print("RL @ %.4g kHz, %d m : Sv=%.1f dB, alpha=%.2f dB/km -> RL=%.1f dB"
              % (f, r, sv_mid, alpha, rl))
    with open(os.path.join(outdir, "resume_rl.json"), "w") as fh:
        json.dump(examples, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=["apl_uw", "searay"], default="apl_uw",
                    help="apl_uw (defaut) ou searay (simplifie Sp + 7log f)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--config", default=None,
                    help="fichier JSON de configuration (fusionne sur CONFIG)")
    ap.add_argument("--freqs", default=None, metavar="K1,K2,...",
                    help="frequences en kHz, ex. '0.1,12,1000' (plage 0.1-1000 kHz)")
    ap.add_argument("--freqs-log", nargs=3, type=float, default=None,
                    metavar=("KMIN", "KMAX", "N"),
                    help="grille logarithmique de KMIN a KMAX kHz, N points")
    ap.add_argument("--hours", default=None, metavar="H1,H2,...",
                    help="heures UTC, ex. '2,14'")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--chl", default=None, metavar="emodnet|erddap|FICHIER",
                    help="source de chlorophylle (unique) : 'emodnet' "
                         "(DIVAnd Mediterranee), 'erddap' (MODIS Aqua mondial), "
                         "un CSV deja telecharge, ou rien (proxy synthetique)")
    ap.add_argument("--bathy", default=None, metavar="FICHIER|auto",
                    help="GeoTIFF EMODnet, 'auto' pour le telecharger via WCS, "
                         "ou rien pour la bathymetrie synthetique")
    args = ap.parse_args()
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        with open(args.config) as fh:
            deep_update(cfg, json.load(fh))
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.freqs_log:
        kmin, kmax, n = args.freqs_log
        freqs = list(np.logspace(math.log10(kmin), math.log10(kmax), int(n)))
    elif args.freqs:
        freqs = [float(x) for x in args.freqs.split(",")]
    else:
        freqs = None
    hours = [int(h) for h in args.hours.split(",")] if args.hours else None
    chl_src = args.chl or cfg["chlorophyll"]["source"]
    if chl_src == "none":
        chl_src = None
    bathy_src = args.bathy or cfg["bathymetry"]["source"]
    main(cfg, args.modele, chl_src=chl_src, bathy_src=bathy_src,
         freqs=freqs, hours=hours, date_str=args.date)
