# reverb-gen

Génération de cartes 3D (lat, lon, immersion, temps) de la réverbération
volumique `S_v` (dB re 1 m⁻¹) pour sonar actif, modèle type APL-UW
(résonance de vessies natatoires + migration nycthémérale de la couche de
diffusion profonde, DSL) + niveau de réverbération RL selon Urick.

**Plage de fréquences supportée : 100 Hz (0.1 kHz) – 1 MHz (1000 kHz).**

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# démo (bathymetrie synthétique), fréquences par défaut 0.1-1000 kHz
python reverberation_3d.py --outdir out

# pleine bande 100 Hz - 1 MHz, grille log
python reverberation_3d.py --freqs-log 0.1 1000 13 --hours 2,14

# fréquences explicites
python reverberation_3d.py --freqs 0.1,12,1000 --hours 2,14

# vraies données : DTM EMODnet (WCS) + chlorophylle
python reverberation_3d.py --bathy auto --chl emodnet
python reverberation_3d.py --bathy auto --chl erddap
python reverberation_3d.py --chl mon_fichier.csv

# modèle simplifié SEARAY (Sv = Sp + 7 log10 f)
python reverberation_3d.py --modele searay

# configuration externe (fusionnée sur CONFIG)
python reverberation_3d.py --config config.example.json
```

## Carte de zone + affichage : `zone_map.py`

Génère la carte S_v d'une zone prédéfinie (ou `--bbox`) puis produit les
figures dans `<outdir>/figures/` : cartes horizontales `map_*.png`
(isobathes 200/1000 m en overlay) et coupes verticales `transect_*.png`.

```bash
# zone Toulon-Calvi, vraie bathymetrie EMODnet + chlorophylle DIVAnd
python zone_map.py --zone toulon_calvi --bathy auto --chl emodnet

# demo rapide sans telechargement (bathy synthetique, Golfe du Lion)
python zone_map.py --zone golfe_du_lion --freqs 0.1,12,1000 --hours 2,14

# re-affichage seul a partir du netCDF existant
python zone_map.py --zone toulon_calvi --no-run --transect toulon calvi

# zone libre + options d'affichage
python zone_map.py --bbox 42.4 43.3 5.8 9.0 --transect toulon calvi \
    --slice-freq 12 --slice-depths 0,120,400 --slice-hours 2,14
```

Zones : `golfe_du_lion`, `toulon_calvi`. Points nommés pour `--transect` :
`toulon`, `calvi`, `marseille`, `nice`, `ajaccio`, `banyuls`.


Options : `--modele apl_uw|searay`, `--outdir`, `--config`, `--freqs`,
`--freqs-log KMIN KMAX N`, `--hours`, `--date`, `--chl emodnet|erddap|CSV`,
`--bathy auto|FICHIER.tif`. Les fréquences hors [0.1, 1000] kHz sont
rejetées (`validate_freqs`).

## Sorties (dans `--outdir`, défaut `out/`)

- `reverb_volume_3d.nc` — netCDF CF-1.6 : `Sv(time, frequency, depth, lat, lon)`
  en dB re 1 m⁻¹ (NaN sous le fond), + `chlorophyll` si `--chl`, + `bathymetry`
- `geotiff/Sv_f*kHz_z* m_*hUTC.tif` — tranches horizontales WGS84 float32
  (ligne 0 = nord), nodata `nan`
- `quicklook.png` — profils jour/nuit + carte + chlorophylle
- `resume_rl.json` — exemples RL (Urick) à 0.1, 6.5, 12, 100, 1000 kHz

## Données externes

- **Bathymetrie** : DTM EMODnet via WCS public (`--bathy auto`, maille ~115 m),
  ou GeoTIFF réutilisable, ou synthétique (démo).
- **Chlorophylle** (source unique) : EMODnet DIVAnd saisonnier Méditerranée
  (profils 0–600 m, ERDDAP) ou MODIS Aqua L3 via ERDDAP CoastWatch
  (climatologie du mois, surface, extension verticale exponentielle).
- **SPM / turbidité** (pistes pour un futur terme d'atténuation, non intégré) :
  CoastWatch ERDDAP `noaacwNPPN20S3AspmSCIDINEOF2kmDaily` (SPM global 2 km),
  EMODnet Physics `TSM_MBSEA` (Méditerranée), Copernicus Marine Service
  `OCEANCOLOUR_MED_BGC_HR_*` (Sentinel-2 MSI 100 m en bande côtière 20 km :
  TUR FNU, SPM mg/l, BBP) et OLCI 300 m.

## Physique et références

- Urick 1983, *Principles of Underwater Sound* : RL = SL − 2TL + Sv + 10 log V
- Jackson et al. 1994, APL-UW TR 9407 : modèles environnementaux HF
- Love 1971 / 1978 : résonance et target strength des vessies natatoires
- Francois & Garrison 1982, JASA 72(6):1879 : absorption (valable ~0.1 kHz–1 MHz ;
  au-delà de ~200 kHz le terme MgSO₄ peut être sous-estimé)
- Mackenzie : célérité ; NOAA : position solaire ; Klevjer et al. 2016,
  Ariza et al. 2016 : profondeur de la DSL
- SEARAY simplifié : Sv = Sp + 7 log10 f

## Limites connues

- Compression de la DSL sur le plateau et bornes jour/nuit (400/120 m) :
  paramétrisations de calibration, pas des équations publiées (Klevjer 2016).
- Chlorophylle satellite = surface uniquement ; extension verticale
  exponentielle paramétrique.
- **Pas de terme SPM** (Richards, Heathershaw & Thorne 1996, JASA
  100(3):1447–1450 : atténuation par les particules en suspension, ~3 dB/100 m
  à 100 kHz pour 0,2 kg/m³) ni dans le TL ni dans le Sv. À 6,5–12 kHz le Sv
  particulaire est négligeable (régime de Rayleigh, σ ∝ f⁴, ~−130 dB pour
  0,2 kg/m³ de limon), mais en eau turbide à ≥ 100 kHz il faudrait l'ajouter
  aux deux (atténuation + backscatter, même section efficace).
- Absorption évaluée à la profondeur = portée (approximation).

### Téléchargements : cache, hors-ligne, erreurs réseau

- Les téléchargements sont en cache (`bathy_*.tif`, `chlorophyll_*.csv`) dans
  `--outdir`, ou d'abord dans `--data-dir` si fourni. Aucun proxy n'est
  utilisé : connexions directes uniquement.
- `--timeout 180 --retries 2` : réglages des téléchargements (défaut 180 s, 2 essais).
- En-têtes HTTP : `User-Agent` descriptif par défaut
  (`reverb-gen/1.0 (+https://github.com/seagnalvibe/reverb-gen)`), surchargeable
  avec `--user-agent 'mon-projet/1.0 (contact@ex.org)'` — certains serveurs de
  données ouverts exigent un UA non standard : identifiez votre client.
  `Accept` est adapté au contenu (`image/tiff` pour le WCS, `text/csv` pour ERDDAP).
- `--offline` : aucun appel réseau ; exige les fichiers locaux (`--chl FICHIER.csv`,
  `--bathy FICHIER.tif`, ou présents dans `--data-dir`/`outdir`). Si un fichier
  manque, l'erreur imprime l'URL exacte à télécharger manuellement sur un poste
  autorisé, puis à rejouer en local.
- En cas d'échec réseau (poste sans accès direct), même principe : l'URL est
  imprimée dans le message d'erreur, télécharge-la à la main, puis relance avec
  le chemin local.

## Tests

```bash
python -m unittest discover -s tests -v
```
