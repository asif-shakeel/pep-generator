#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pep_generator.py — CLEAN, FULL VERSION (every agent transitions & emits each bin)

Matrices are column-stochastic: columns = origins j, rows = destinations i.

What changed vs the “old file”
- Removed all “active subset” / roster logic. Now **every agent** transitions at every bin,
  and **every transition** is emitted as a row. TOTAL_PER_BIN_FIXED = total # of agents.
- Kept: region filtering, λ AM/MIDDAY/PM with single g(r) (km), multinomial assignment,
  initial mass modes, initial state modes (incl. periodic fixed point), transitions+manifest,
  validate() with bin-based diagnostics: pep_temporal_diagnostics.csv, pep_mean_bin_balance.csv,
  pep_daily_balance.csv.
"""

from __future__ import annotations
import os
import json
import math
import gzip
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# =========================
# Mode
# =========================
MODE = "generate"  # {"generate","validate"}

# =========================
# Paths & template
# =========================
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "/Users/asif/Documents/nm24/data")
OUTPUT_DIR = os.path.join(BASE_DIR, "/Users/asif/Documents/nm24/outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "/Users/asif/Documents/nm24/plots")
TRANS_DIR  = os.path.join(OUTPUT_DIR, "/Users/asif/Documents/nm24/transitions")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Template discovery (as in original flow)
TEMPLATE_FILENAME = "od_mx_agg5_3h.csv"
TEMPLATE_PATH     = os.path.join(DATA_DIR, TEMPLATE_FILENAME)
USE_TEMPLATE      = True

# Column names (template & outputs)
START_COL = "start_geohash5"
END_COL   = "end_geohash5"
DATE_COL  = "local_date"
TIME_COL  = "local_time"
COUNT_COL = "trip_count"

# =========================
# Sweep window & time resolution
# =========================
SWEEP_START_DATE = "2019-07-01"   # inclusive
SWEEP_END_DATE   = "2019-09-30"   # inclusive
WINDOW_START_HH  = 0               # inclusive
WINDOW_END_HH    = 24              # exclusive
TIME_RES_MIN     = 45              # bin size (minutes)

# =========================
# Files / plots / randomness (RESTORED)
# =========================
WRITE_GZIP             = True  # False #   # legacy name; maps to COMPRESS_OUTPUTS
WRITE_RAW              = False
WRITE_OD_AGGREGATE     = True
SAVE_TRANSITIONS       = False
VALIDATE_ON_GENERATE   = True
MAKE_PLOTS             = False
PLOT_TOP_ORIGINS       = 4
PLOT_TOP_DEST_PER_ORG  = 10
SEED                   = 1234    # restored default

# Internal mapping for compression
COMPRESS_OUTPUTS = WRITE_GZIP

# =========================
# Geohash precision (fixed 5)
# =========================
GEOHASH_PRECISION = 5

# =========================
# Region filter (RESTORED & WIRED)
# =========================
COUNTRY = "Mexico"
REGION_NAME = "Mexico City, Mexico"
REGION_BBOXES = {
    "Mexico City, Mexico": (-99.45465164940843, 18.96381286042833, -98.85051749282216, 19.677365270912603),
}
REGION_ENDPOINT_RULE = "either"     # {"both","either"}
BUFFER_KM_BY_COUNTRY = {"Colombia":10.0, "Indonesia":10.0, "Mexico":10.0, "India":10.0}
USE_REGION_FILTER = True
REGION_MEMBERSHIP_MODE = "intersects"  # {"intersects","within"}; for points, both behave the same



# =========================
# Gravity & distance model (RESTORED semantics)
# =========================
ALPHA          = 0.70    # mass exponent (maps to ALPHA)
BETA           = 2.8     # power distance decay exponent
USE_EXP_DECAY  = False   # if True, use exp(-d / DIST_SCALE_KM)
DIST_SCALE_KM  = 5.0     # only used when USE_EXP_DECAY=True ⇒ gamma = 1/DIST_SCALE_KM
HARD_CUTOFF_KM = None    # if not None, neighbors with d > cutoff have 0 weight (except self)

# Derived internal names
GAMMA_DECAY = 0.0 if not USE_EXP_DECAY else (0.0 if DIST_SCALE_KM in (None, 0) else 1.0/float(DIST_SCALE_KM))
DIST_DECAY_MODE = "exp" if USE_EXP_DECAY else "power"

# =========================
# Center, Neighbor scheme & parameters
# =========================
CENTER_K = 6
NEIGHBOR_SCHEME     = "lattice"   # {"knn","lattice"}
NEIGHBOR_K          = 25          # hard cap
NEIGHBOR_INNER_K    = 9          # ring-1 threshold for inner boost & hop labels
NEIGHBOR_BOOST      = 8.0         # multiplier for inner neighbors (excluding self)
NEIGHBOR_MAX_RINGS  = 2           # lattice mode: max rings to include

# --- Optional neighbor diagnostics (counts of destinations used per origin & bin) ---
ENABLE_NEIGHBOR_DIAGNOSTICS = False  # default False to preserve current behavior


# =========================
# Distance sampling (ring-aware truncated lognormal)
# =========================
CELL_EDGE_KM_EST   = 4.9
CELL_DIAG_KM_EST   = CELL_EDGE_KM_EST * math.sqrt(2.0)
SELF_MAX_PERIM_KM  = 4.0 * CELL_EDGE_KM_EST
R1_MAX_KM          = CELL_DIAG_KM_EST
R2_MIN_KM          = 0.75 * CELL_EDGE_KM_EST
R2_MAX_KM          = 2.0 * CELL_DIAG_KM_EST
LN_MEDIANS = {0: 0.25, 1: 0.9, 2: 1.8}
LN_SIGMA   = 0.55

# --- Optional auto-median tuning for hop-distance sampling (default OFF; backward compatible) ---
ENABLE_AUTO_DISTANCE_MEDIANS = False          # leave False to preserve exact current behavior
AUTO_DISTANCE_MEDIAN_STRATEGY = "geom_bounds" # {"geom_bounds","empirical_neighbors"}
AUTO_LN_SIGMA = None                          # if set (e.g., 0.35), overrides LN_SIGMA when auto is enabled
ENABLE_UNIFORM_DISTANCE_SAMPLING = False

# --- Pairwise distance envelopes for LATTICE mode (optional; default OFF) ---
ENABLE_PAIRWISE_DISTANCE_ENVELOPES = True   # preserves current behavior when False


# Self-loop envelope convention:
# - "single_diag": max = sqrt(2) * CELL_EDGE_KM_EST
# - "double_diag": max = 2 * sqrt(2) * CELL_EDGE_KM_EST (allows there-and-back in a bin)
SELF_DISTANCE_MODE = "single_diag"           # {"single_diag","double_diag"}

# (Advanced) Persist the precomputed envelopes for transparency/debugging
DUMP_PAIRWISE_ENVELOPES_JSON = False




# --- Diagnostics toggles (default ON to keep current behavior) ---
ENABLE_DIAG_MODEL_EMP_COMPARE = False   # pep_model_empirical_compare.csv
ENABLE_DIAG_TEMPORAL          = False   # pep_temporal_diagnostics.csv
ENABLE_DIAG_MEAN_BIN          = True   # pep_mean_bin_balance.csv
ENABLE_DIAG_DAILY             = True   # pep_daily_balance.csv


# =========================
# Initial mass & initial state configuration
# =========================
# Initial mass for destinations (hourly base BEFORE Markov normalization)
MASS_INIT_MODE     = "flat" #"template_window_mean"  # {"flat","template_inflow_day0","template_window_mean"}
MASS_INIT_HOUR_SET = None  # optional: restrict window mean to these hours (set[int])

# Initial state (starting distribution over origins for agents)
INIT_X_MODE = "flat" # "template_inflow_day0"  # {"flat","template_inflow_day0","template_window_mean","stationary_meanP","periodic_fixed_point"}

# --- Radial initialization (optional; defaults OFF to preserve behavior) ---
RADIAL_INIT_ENABLE_MASS = False     # If True, reshape initial masses by a radial profile
RADIAL_INIT_ENABLE_X    = False     # If True, reshape initial state x by a radial profile

# Piecewise profile parameters (km)
RADIAL_INIT_INNER_KM = 10.0   # flat "center" plateau radius
RADIAL_INIT_OUTER_KM = 35.0   # flat "periphery" plateau radius

# Multipliers at plateaus; linear interpolation between inner and outer
# Example: center heavier than periphery -> (1.4, 0.8) ; for the opposite, flip them.
RADIAL_INIT_CENTER_MULT  = 1.0
RADIAL_INIT_PERIPH_MULT  = 1.0

# Normalization choice after applying radial multipliers
# "mean1" → rescale vector to have mean 1 (for masses) or sum 1 (for x, done separately)
# "none"  → no rescale (rarely useful)
RADIAL_INIT_NORMALIZE = "mean1"   # {"mean1","none"}



# --- Per-bin population control ---
# {"fixed", "template_mean", "template_mean_day0"}
TOTAL_PER_BIN_MODE  = "fixed"
TOTAL_PER_BIN_FIXED = 12000  # used when mode == "fixed"



# --- Auto-tune options for saturation radii (center/periphery) ---
AUTO_TUNE_SAT_RADII = True          # enable auto tuning
AUTO_TUNE_MODE      = "floor"       # {"floor","replace"}
AUTO_PERIPH_Q       = 0.92          # percentile of all r_km for periphery plateau
AUTO_CENTER_PAD_KM  = 0.5 * CELL_EDGE_KM_EST  # cushion added beyond furthest center

# Optional hard overrides (None = unused)
RC_OVERRIDE_KM = None
RP_OVERRIDE_KM = None


# --- Temporal ramp sets for lambda modulation ---
AM_HOURS_SET     = {6, 7, 8, 9, 10}           # morning inflow ramp
MIDDAY_HOURS_SET = {11, 12, 13}   # midday plateau
PM_HOURS_SET     = {14, 15, 16, 17, 18, 19, 20, 21}  # evening outflow ramp

# =========================
# Stays
# =========================
P_STAY_BASE = 0.80   # baseline stickiness for all tiles

# =========================
# λ-based AM/PM modulation with single g(r) in KM
# =========================



LAMBDA_DEST = 0.35   # controls time-varying destination attraction
LAMBDA_STAY = 0.05   # small retention slope (optional)

R_CENTER_SAT_KM = 10.0   # r_c: center plateau radius (km)
R_PERIPH_SAT_KM = 35.0   # r_p: periphery plateau radius (km)



# Optional diagnostics-only bucketing for reports (does not affect g(r))
DIAGNOSTIC_RING_BINS = 8





# TWEAKS

# =========================
# TEMPORAL MODULATION & HOURS
# =========================
AM_HOURS_SET     = {6, 7, 8, 9, 10}               # AM inflow
MIDDAY_HOURS_SET = {11, 12, 13}                   # midday flat
PM_HOURS_SET     = {14, 15, 16, 17, 18, 19, 20, 21}  # PM outflow

# =========================
# STAY BEHAVIOR (no hourly multipliers)
# =========================
P_STAY_BASE = 0.80
P_STAY_HOURLY_MULT = {}
P_STAY_CENTER_BY_HOUR = {}
P_STAY_PERIPH_BY_HOUR = {}

# =========================
# LAMBDA MODULATION (core of temporal dynamics)
# =========================
LAMBDA_DEST = 0.35   # destination attraction: center pull in AM, periphery pull in PM
LAMBDA_STAY = 0.10   # stay modulation: center sticky in AM, periphery sticky in PM

# =========================
# RADIAL INITIALIZATION (periphery-heavy start)
# =========================
 
RADIAL_INIT_ENABLE_MASS = True
RADIAL_INIT_ENABLE_X    = True
RADIAL_INIT_INNER_KM    = 10.0
RADIAL_INIT_OUTER_KM    = 35.0
RADIAL_INIT_CENTER_MULT = 0.85
RADIAL_INIT_PERIPH_MULT = 1.20
RADIAL_INIT_NORMALIZE   = "mean1"

# =========================
# AUTO-TUNE SATURATION RADII (affects g(r) profile)
# =========================
AUTO_TUNE_SAT_RADII = True
AUTO_TUNE_MODE      = "floor"
AUTO_PERIPH_Q       = 0.90
AUTO_CENTER_PAD_KM  = 2.0
RC_OVERRIDE_KM      = None
RP_OVERRIDE_KM      = None

# =========================
# NEIGHBORHOOD CONFIG
# =========================
NEIGHBOR_SCHEME    = "lattice"  # or "knn"
NEIGHBOR_K         = 25
NEIGHBOR_INNER_K   = 9
NEIGHBOR_MAX_RINGS = 2

CELL_EDGE_KM_EST = 1.2

# =========================
# TOTAL AGENTS & BIN CONFIG
# =========================
TOTAL_PER_BIN_MODE   = "fixed"
TOTAL_PER_BIN_FIXED  = 12000




# =========================
# Logging helpers
# =========================
def banner(msg: str) -> None:
    print("\n" + "="*28 + f"\n{msg}\n" + "="*28, flush=True)

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

# =========================
# Time helpers & λ schedule
# =========================
def build_time_bins_for_window(d_start: str, d_end: str, hh_start: int, hh_end: int, res_min: int) -> List[pd.Timestamp]:
    ds = pd.to_datetime(d_start)
    de = pd.to_datetime(d_end)
    out = []
    day = ds
    while day <= de:
        t0 = day + pd.Timedelta(hours=hh_start)
        t1 = day + pd.Timedelta(hours=hh_end)
        t = t0
        while t < t1:
            out.append(t)
            t += pd.Timedelta(minutes=res_min)
        day += pd.Timedelta(days=1)
    return out

def _hour_bounds_from_set(H: set[int]) -> Tuple[Optional[int],Optional[int]]:
    if not H:
        return None, None
    return int(min(H)), int(max(H))

def phase_s_of_minute(minute_of_day: int) -> float:
    """
    Returns a piecewise phase s(t) in {-1, 0, +1}:
      AM_HOURS_SET  -> +1  (pull to center)
      MIDDAY_HOURS_SET -> 0  (flat)
      PM_HOURS_SET  -> -1  (push to periphery)
      otherwise -> 0  (night: rest)
    """
    am0, am1 = _hour_bounds_from_set(AM_HOURS_SET)
    md0, md1 = _hour_bounds_from_set(MIDDAY_HOURS_SET)
    pm0, pm1 = _hour_bounds_from_set(PM_HOURS_SET)

    m = minute_of_day
    h = m // 60

    if am0 is not None and am0 <= h <= am1:
        return +1.0
    if md0 is not None and md0 <= h <= md1:
        return 0.0
    if pm0 is not None and pm0 <= h <= pm1:
        return -1.0
    return 0.0

# =========================
# Geospatial helpers
# =========================
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BITS = np.array([16,8,4,2,1])

def geohash_decode(g: str) -> Tuple[float,float]:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    even = True
    for ch in g:
        cd = _BASE32.find(ch)
        if cd == -1:
            raise ValueError(f"Invalid geohash char: {ch}")
        for mask in _BITS:
            bit = 1 if (cd & mask) else 0
            if even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            even = not even
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return float(lat), float(lon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Region filter helpers
def _buffer_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float, buffer_km: float) -> Tuple[float,float,float,float]:
    if buffer_km in (None, 0.0):
        return lon_min, lat_min, lon_max, lat_max
    lat0 = 0.5 * (lat_min + lat_max)
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * max(1e-6, math.cos(math.radians(lat0))))
    return (lon_min - dlon, lat_min - dlat, lon_max + dlon, lat_max + dlat)

def _point_in_bbox(lon: float, lat: float, bbox: Tuple[float,float,float,float]) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 <= lon <= x1) and (y0 <= lat <= y1)

# =========================
# Template preprocessing + region filtering
# =========================
def window_slice_template(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def apply_region_filter(df: pd.DataFrame, region_name: Optional[str]) -> pd.DataFrame:
    if not USE_REGION_FILTER or not region_name:
        return df
    if region_name not in REGION_BBOXES:
        warn(f"Region '{region_name}' not in REGION_BBOXES; skipping region filter.")
        return df

    bbox = REGION_BBOXES[region_name]
    buf_km = BUFFER_KM_BY_COUNTRY.get(COUNTRY, 0.0)
    bbox_b = _buffer_bbox(*bbox, buffer_km=buf_km)

    # Decode geohashes to centers
    uniq = pd.unique(pd.concat([df[START_COL], df[END_COL]], ignore_index=True))
    gh2pt = {}
    for g in uniq:
        try:
            lat, lon = geohash_decode(str(g))
            gh2pt[str(g)] = (lon, lat)
        except Exception:
            gh2pt[str(g)] = (None, None)

    def in_region(g: str) -> bool:
        lonlat = gh2pt.get(str(g), (None, None))
        if lonlat[0] is None:
            return False
        inside = _point_in_bbox(lonlat[0], lonlat[1], bbox_b)
        if REGION_MEMBERSHIP_MODE == "within":
            return inside
        return inside

    start_in = df[START_COL].map(in_region)
    end_in   = df[END_COL].map(in_region)

    if REGION_ENDPOINT_RULE == "both":
        mask = start_in & end_in
    else:
        mask = start_in | end_in

    out = df.loc[mask].copy()
    info(f"Region filter '{region_name}': kept {len(out)}/{len(df)} rows (buffer {buf_km} km)")
    return out

# =========================
# Centers & radial distances (km)
# =========================
def infer_centers_by_inflow(df: pd.DataFrame, k: int) -> List[str]:
    if COUNT_COL not in df.columns:
        s = df[END_COL].value_counts()
    else:
        s = df.groupby(END_COL)[COUNT_COL].sum().sort_values(ascending=False)
    return list(s.head(k).index.astype(str))

def centroid_of_tiles(tiles: List[str]) -> Tuple[float,float]:
    if not tiles:
        return (0.0, 0.0)
    latlons = np.array([geohash_decode(t) for t in tiles], dtype=float)
    return float(latlons[:,0].mean()), float(latlons[:,1].mean())

def radial_km(nodes: List[str], center_latlon: Tuple[float,float]) -> Dict[str, float]:
    clat, clon = center_latlon
    out = {}
    for g in nodes:
        lat, lon = geohash_decode(g)
        out[g] = float(haversine_km(clat, clon, lat, lon))
    return out

# Single radial attenuation g(r) with km-based saturation
def g_of_r(r_km: float, r_center_km: float, r_periph_km: float) -> float:
    rc, rp = float(r_center_km), float(r_periph_km)
    if rp <= rc:
        rp = rc + 1e-6
    u = (r_km - rc) / (rp - rc)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return float(1.0 - u)  # g ∈ [0,1]

def diagnostic_ring_labels(nodes: List[str], rkm_map: Dict[str,float], n_bins: int) -> Dict[str,int]:
    if n_bins <= 0:
        return {g: 0 for g in nodes}
    rvals = np.array([rkm_map[g] for g in nodes], dtype=float)
    qs = np.quantile(rvals, np.linspace(0,1,n_bins+1))
    bins = np.digitize(rvals, qs[1:-1], right=True)
    return {g:int(b) for g,b in zip(nodes, bins)}

# =========================
# Neighbor graph
# =========================
import numpy as np
from math import sqrt

def _lattice_neighbor_indices(
    nodes: List[str],
    K_cap: int,
    max_rings: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    True lattice neighborhood: within ±max_rings grid steps (Chebyshev metric).
    Returns (nb_idx, nb_dist) with self at rank 0 and neighbors sorted by ring, then Euclidean distance.

    Notes:
      - Uses local equirectangular projection to get x/y in km for snap-to-grid and for distance calc.
      - Applies K_cap (truncate if (2*max_rings+1)^2 > K_cap).
      - Pads rows (edge/border nodes) by repeating self with distance 0 so the output is rectangular.
    """
    N = len(nodes)
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    # Decode all geohashes once (lat, lon)
    latlon = np.array([geohash_decode(g) for g in nodes], dtype=float)  # [:,0]=lat, [:,1]=lon

    # Local equirectangular projection to km, relative to mean lat/lon (stable numerically)
    lat_ref = float(latlon[:, 0].mean())
    lon_ref = float(latlon[:, 1].mean())
    def _to_xy_km(lat: float, lon: float) -> Tuple[float, float]:
        # reuse your helper; avoids drift with small areas
        dx_km, dy_km = _equirect_km_offsets(lat_ref, lon_ref, lat, lon)
        return dx_km, dy_km

    centers_xy = np.array([_to_xy_km(float(lat), float(lon)) for lat, lon in latlon], dtype=float)  # (N,2)

    # Snap to grid using estimated cell edge
    e_km = float(CELL_EDGE_KM_EST)
    gx = np.round(centers_xy[:, 0] / e_km).astype(int)
    gy = np.round(centers_xy[:, 1] / e_km).astype(int)

    # Build grid -> index map (assumes one node per cell; if collisions exist, first wins)
    grid = {(int(gx[i]), int(gy[i])): i for i in range(N)}

    # Precompute sorted offsets by ring (Chebyshev radius), then Euclidean
    offsets: List[Tuple[int,int,int,float]] = []  # (dx, dy, cheb, euclid)
    for dx in range(-max_rings, max_rings + 1):
        for dy in range(-max_rings, max_rings + 1):
            cheb = max(abs(dx), abs(dy))
            eu   = math.hypot(dx, dy)
            offsets.append((dx, dy, cheb, eu))
    # Self should be first (cheb=0, eu=0)
    offsets.sort(key=lambda t: (t[2], t[3]))  # (chebyshev ring, euclidean within ring)

    # Construct neighbor lists per origin
    neigh_lists: List[List[int]] = []
    dist_lists : List[List[float]] = []

    for i in range(N):
        cx, cy = int(gx[i]), int(gy[i])
        lst_idx: List[int] = []
        lst_dst: List[float] = []

        for dx, dy, _, _ in offsets:
            cell = (cx + dx, cy + dy)
            j = grid.get(cell)
            if j is None:
                continue
            # append neighbor
            lst_idx.append(j)
            # distance in km between centers (use haversine to stay consistent with power/exp kernels)
            d_km = haversine_km(
                float(latlon[i, 0]), float(latlon[i, 1]),
                float(latlon[j, 0]), float(latlon[j, 1])
            )
            lst_dst.append(float(d_km))

        # Ensure self is present at rank 0 (it will be, since dx=dy=0 is in offsets, but double-safeguard)
        if not lst_idx or lst_idx[0] != i:
            # Put self at front if missing
            if i in lst_idx:
                pos = lst_idx.index(i)
                # move self to front, distance 0
                lst_idx.pop(pos)
                lst_dst.pop(pos)
            lst_idx.insert(0, i)
            lst_dst.insert(0, 0.0)

        # Apply K cap
        if K_cap is not None and K_cap > 0:
            lst_idx = lst_idx[:int(K_cap)]
            lst_dst = lst_dst[:int(K_cap)]

        neigh_lists.append(lst_idx)
        dist_lists.append(lst_dst)

    # Make rectangular: pad with self + 0 distance if row shorter than max_len
    K_eff = max(len(x) for x in neigh_lists)
    nb_idx  = np.full((N, K_eff), -1, dtype=int)
    nb_dist = np.zeros((N, K_eff), dtype=float)

    for i in range(N):
        row_idx = neigh_lists[i]
        row_dst = dist_lists[i]
        # pad
        if len(row_idx) < K_eff:
            pad_len = K_eff - len(row_idx)
            row_idx = row_idx + [i] * pad_len
            row_dst = row_dst + [0.0] * pad_len
        nb_idx[i, :]  = np.array(row_idx, dtype=int)
        nb_dist[i, :] = np.array(row_dst, dtype=float)

    return nb_idx, nb_dist


def _knn_neighbor_indices(nodes: List[str], K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    KNN neighbors (including self) for each node.

    Returns:
        nb_idx  : (N, K_eff) int array of neighbor indices per origin (sorted by distance asc)
        nb_dist : (N, K_eff) float array of corresponding distances (km)
    Notes:
        - K_eff = min(K, N) to avoid argpartition errors when K > N.
        - Ensures self is present in each neighbor list; if not, it replaces the farthest entry.
    """
    N = len(nodes)
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    K_eff = max(1, min(int(K), N))  # defensive clamp

    # Decode all geohashes once
    latlons = np.array([geohash_decode(g) for g in nodes], dtype=float)

    # Full symmetric distance matrix (km)
    dmat = np.zeros((N, N), dtype=float)
    for j in range(N):
        latj, lonj = latlons[j]
        for i in range(j + 1, N):
            lati, loni = latlons[i]
            d = haversine_km(latj, lonj, lati, loni)
            dmat[j, i] = d
            dmat[i, j] = d
    # self-distance = 0 by construction

    # Get K_eff nearest (by column), then transpose to (N, K_eff)
    nb_idx = np.argpartition(dmat, kth=K_eff - 1, axis=0)[:K_eff, :].T
    nb_dist = np.take_along_axis(dmat, nb_idx, axis=1)

    # Ensure self is present; sort by distance
    for j in range(N):
        row_idx = nb_idx[j]
        row_dist = nb_dist[j]

        # If self not present, replace farthest with self and distance=0
        if j not in row_idx:
            far = int(np.argmax(row_dist))
            row_idx[far] = j
            row_dist[far] = 0.0

        # Sort this row by distance ascending
        order = np.argsort(row_dist)
        nb_idx[j] = row_idx[order]
        nb_dist[j] = row_dist[order]

    return nb_idx.astype(int), nb_dist.astype(float)




def neighbor_indices(nodes: List[str], K: int, scheme: str, max_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    scheme = (scheme or "knn").lower()
    if scheme == "lattice":
        return _lattice_neighbor_indices(nodes, K_cap=int(K), max_rings=int(max_rings))

    # --- KNN with optional ring-based cutoff ---
    nb_idx, nb_dist = _knn_neighbor_indices(nodes, int(K))

    # apply cutoff if max_rings is given
    if max_rings is not None and int(max_rings) > 0:
        cutoff_km = float(max_rings) * float(CELL_EDGE_KM_EST)
        N, K_eff = nb_idx.shape
        for j in range(N):
            keep = (nb_idx[j] == j) | (nb_dist[j] <= cutoff_km)  # always keep self
            if np.all(keep):
                continue
            # compact row while preserving order
            row_idx = nb_idx[j][keep]
            row_dst = nb_dist[j][keep]
            # pad to K_eff with self
            pad = K_eff - row_idx.size
            if pad > 0:
                row_idx = np.concatenate([row_idx, np.full(pad, j, dtype=int)])
                row_dst = np.concatenate([row_dst, np.zeros(pad, dtype=float)])
            nb_idx[j]  = row_idx[:K_eff]
            nb_dist[j] = row_dst[:K_eff]

    return nb_idx, nb_dist




# =========================
# Day pattern: apply λ using g(r_km) and s(t)
# =========================
def apply_bias_to_masses(nodes: List[str], rkm_map: Dict[str,float], M_base: np.ndarray, minute_of_day: int,
                         lambda_dest: float) -> np.ndarray:
    s = phase_s_of_minute(minute_of_day)
    if abs(s) < 1e-12 or lambda_dest == 0.0:
        return M_base.copy()
    gvals = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)
    factor = 1.0 + lambda_dest * s * gvals  # g ∈ [0,1], s ∈ [-1,1]
    factor = np.clip(factor, 1e-6, None)
    return M_base * factor

# =========================
# Markov builder
# =========================
def build_P_for_minute(nodes: List[str], rkm_map: Dict[str,float], centers_set: set, minute_of_day: int,
                       M_base: np.ndarray,
                       nb_idx: np.ndarray, nb_dist: np.ndarray) -> np.ndarray:
    N, K = nb_idx.shape
    M_mod = apply_bias_to_masses(nodes, rkm_map, M_base, minute_of_day, LAMBDA_DEST)
    Mw = np.power(M_mod, ALPHA)

    h = (minute_of_day // 60) % 24
    P = np.zeros((N, N), dtype=float)

    gvals = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)
    s = phase_s_of_minute(minute_of_day)

    periph_set = {g for g in nodes if rkm_map[g] >= R_PERIPH_SAT_KM}

    for j in range(N):
        g_j = gvals[j]            # g≈1 at center, g≈0 at periphery
        s = phase_s_of_minute(minute_of_day)

        # λ-driven stay modulation:
        #   AM (s=+1):   center ↑stay, periphery ↓stay
        #   MIDDAY (s=0): neutral
        #   PM (s=-1):   center ↓stay, periphery ↑stay
        center_vs_periph = (2.0 * g_j - 1.0)  # +1 center ... -1 periphery
        pst = P_STAY_BASE * (1.0 + LAMBDA_STAY * s * center_vs_periph)

        # Keep it sane
        pst = float(np.clip(pst, 0.0, 0.98))


        neigh = nb_idx[j]
        distk = nb_dist[j]

        # Hard cutoff (except self)
        if HARD_CUTOFF_KM is not None:
            cutoff_mask = (distk > float(HARD_CUTOFF_KM)) & (neigh != j)
        else:
            cutoff_mask = np.zeros_like(distk, dtype=bool)

        # Gravity weights
        if DIST_DECAY_MODE == "power":
            with np.errstate(divide='ignore'):
                w = np.power(np.maximum(Mw[neigh], 1e-12), 1.0) / np.power(np.maximum(distk, 1e-6), BETA)
        else:
            w = np.power(np.maximum(Mw[neigh], 1e-12), 1.0) * np.exp(-GAMMA_DECAY * distk)

        # Neighbor boost for inner neighbors (excluding self)
        if NEIGHBOR_INNER_K > 0 and NEIGHBOR_BOOST not in (None, 1.0, 0.0):
            inner_mask = np.zeros_like(neigh, dtype=bool)
            inner_mask[1:min(NEIGHBOR_INNER_K, len(inner_mask))] = True
            w = w * np.where(inner_mask & (neigh != j), float(NEIGHBOR_BOOST), 1.0)

        # Apply hard cutoff
        w = np.where(cutoff_mask, 0.0, w)

        # Separate self vs others, normalize
        mask = (neigh != j)
        neigh_noself = neigh[mask]
        w_noself = w[mask]
        ssum = float(np.sum(w_noself))
        if ssum <= 0:
            P[j, j] = 1.0
            continue
        w_noself = w_noself / ssum

        P[j, j] = pst
        spread = 1.0 - pst
        P[neigh_noself, j] += spread * w_noself

        colsum = P[:, j].sum()
        P[:, j] = P[:, j] / colsum if colsum > 0 else np.eye(N, dtype=float)[:, j]

    return P

# =========================
# Distance sampling
# =========================
def _sample_lognormal_km(median_km: float, sigma: float, rng: np.random.Generator) -> float:
    mu = math.log(max(median_km, 1e-9))
    return float(np.exp(rng.normal(mu, sigma)))

def _truncate_or_resample(x: float, lo: Optional[float], hi: Optional[float], rng: np.random.Generator,
                          sampler) -> float:
    if lo is None and hi is None:
        return x
    attempts = 0
    while attempts < 64 and ((lo is not None and x < lo) or (hi is not None and x > hi)):
        x = sampler(); attempts += 1
    if lo is not None and x < lo: x = lo
    if hi is not None and x > hi: x = hi
    return x

def sample_hop_distance_km(hop_group: int, rng: np.random.Generator) -> float:

    if ENABLE_UNIFORM_DISTANCE_SAMPLING:
        # new flag (default False)
        if hop_group == 0:
            lo, hi = 0.0, SELF_MAX_PERIM_KM
        elif hop_group == 1:
            lo, hi = CELL_EDGE_KM_EST, R1_MAX_KM
        else:
            lo, hi = R2_MIN_KM, R2_MAX_KM
        return float(rng.uniform(lo, hi))

    med = LN_MEDIANS.get(int(hop_group), LN_MEDIANS[2]); sig = LN_SIGMA
    if hop_group == 0:
        lo, hi = 0.0, SELF_MAX_PERIM_KM
    elif hop_group == 1:
        lo, hi = CELL_EDGE_KM_EST, R1_MAX_KM
    else:
        lo, hi = R2_MIN_KM, R2_MAX_KM
    sampler = lambda: _sample_lognormal_km(med, sig, rng)
    return _truncate_or_resample(sampler(), lo, hi, rng, sampler)

# --- Helpers for pair-specific envelopes in lattice mode (additive) ---

def _self_max_km_from_mode() -> float:
    diag = math.sqrt(2.0) * float(CELL_EDGE_KM_EST)
    return float(2.0 * diag) if SELF_DISTANCE_MODE == "double_diag" else float(diag)

def _equirect_km_offsets(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    """
    Local small-area approximation to convert (lat,lon) delta to km in x (east) and y (north).
    """
    # mean latitude for scale
    phi = math.radians((lat0 + lat1) * 0.5)
    kx = 111.320 * math.cos(phi)   # km per degree lon
    ky = 110.574                   # km per degree lat
    dx_km = (lon1 - lon0) * kx
    dy_km = (lat1 - lat0) * ky
    return float(dx_km), float(dy_km)

def _abs_round_div(x: float, e: float) -> int:
    """Round |x|/e to nearest integer."""
    return int(round(abs(x) / max(e, 1e-9)))

def _ab_from_centers(lat0: float, lon0: float, lat1: float, lon1: float, e_km: float) -> Tuple[int, int]:
    """
    Infer lattice offsets (a,b) in units of CELL_EDGE_KM_EST from two tile centers using local km axes.
    a = horizontal steps (east/west), b = vertical steps (north/south).
    """
    dx_km, dy_km = _equirect_km_offsets(lat0, lon0, lat1, lon1)
    a = _abs_round_div(dx_km, e_km)
    b = _abs_round_div(dy_km, e_km)
    return a, b

def _dmin_dmax_for_ab(a: int, b: int, e_km: float, self_max_km: float) -> Tuple[float, float]:
    """
    Distance envelope between two axis-aligned squares (edge = e_km) separated by (a,b) steps.
    Formulas:
      d_min = e * sqrt(max(a-1,0)^2 + max(b-1,0)^2)
      d_max = e * sqrt((a+1)^2 + (b+1)^2)
    Self (a=b=0) uses [0, self_max_km] per SELF_DISTANCE_MODE.
    """
    if a == 0 and b == 0:
        return 0.0, float(self_max_km)
    am = max(a - 1, 0)
    bm = max(b - 1, 0)
    dmin = e_km * math.sqrt(am*am + bm*bm)
    dmax = e_km * math.sqrt((a + 1)*(a + 1) + (b + 1)*(b + 1))
    return float(dmin), float(dmax)


def precompute_pairwise_envelopes_generic(nodes, nb_idx):
    N, K = nb_idx.shape
    latlon = np.array([geohash_decode(g) for g in nodes], dtype=float)
    # anchor for local km axes
    lat_ref, lon_ref = float(latlon[:,0].mean()), float(latlon[:,1].mean())

    def to_xy_km(lat, lon):
        return _equirect_km_offsets(lat_ref, lon_ref, lat, lon)

    xy = np.array([to_xy_km(float(lat), float(lon)) for lat, lon in latlon], dtype=float)
    e = float(CELL_EDGE_KM_EST)
    dmin = np.zeros((N, K), dtype=float)
    dmax = np.zeros((N, K), dtype=float)
    self_max = _self_max_km_from_mode()

    for j in range(N):
        xj, yj = xy[j]
        for r in range(K):
            i = int(nb_idx[j, r])
            if i == j:
                dmin[j, r] = 0.0
                dmax[j, r] = self_max
                continue
            xi, yi = xy[i]
            dx = abs(xi - xj); dy = abs(yi - yj)
            # exact box-to-box min, conservative max
            dmin[j, r] = math.hypot(max(dx - e, 0.0), max(dy - e, 0.0))
            dmax[j, r] = math.hypot(dx + e, dy + e)
    return dmin, dmax

def sample_pair_distance_uniform(
    j: int, i: int, nb_idx_row: np.ndarray,
    dmin_row: Optional[np.ndarray], dmax_row: Optional[np.ndarray],
    rng: np.random.Generator
) -> Optional[float]:
    """
    Uniform distance draw using precomputed envelopes aligned with nb_idx.
    Returns None if envelopes are unavailable for the (j,i) pair.
    """
    if dmin_row is None or dmax_row is None:
        return None
    # find neighbor rank r where nb_idx[j, r] == i
    pos = np.where(nb_idx_row == i)[0]
    if pos.size == 0:
        return None
    r = int(pos[0])
    lo = float(dmin_row[r]); hi = float(dmax_row[r])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo):
        return None
    if hi == lo:
        return float(lo)
    return float(rng.uniform(lo, hi))

def _geom_mean(a: float, b: float) -> float:
    a = max(a, 1e-9); b = max(b, 1e-9)
    return float(math.sqrt(a * b))

def _auto_distance_medians_from_bounds() -> Tuple[float, float]:
    """
    Compute medians for hop_group 1 and 2 from the configured annuli.
    Group 1: [CELL_EDGE_KM_EST, R1_MAX_KM]
    Group 2: [R2_MIN_KM,       R2_MAX_KM]
    """
    m1 = _geom_mean(float(CELL_EDGE_KM_EST), float(R1_MAX_KM))
    m2 = _geom_mean(float(R2_MIN_KM),        float(R2_MAX_KM))
    return m1, m2

def _auto_distance_medians_from_neighbors(nb_dist: np.ndarray, inner_k: int) -> Tuple[float, float]:
    """
    Use neighbor geometry:
      - Group 1 = ranks 1..inner_k-1 (exclude self rank 0)
      - Group 2 = ranks >= inner_k
    We take geometric means of the 20th and 80th percentile radii within each group
    to avoid extreme tails, then geometric-mean those bounds for the median.
    """
    # nb_dist shape: (N, K), column j contains distances to neighbors in ascending order
    if nb_dist.ndim != 2 or nb_dist.shape[1] < 2:
        # fallback to bounds if something is off
        return _auto_distance_medians_from_bounds()

    # For all origins, collect distances by rank group:
    # rank 0 is self (0 km); inner ranks are 1..inner_k-1
    K = nb_dist.shape[1]
    inner_hi = max(1, min(inner_k - 1, K - 1))  # ensure at least rank 1 exists
    has_inner = inner_hi >= 1

    # Flatten distances across all origins for each group
    d_inner = nb_dist[:, 1:inner_hi+1].ravel() if has_inner else np.array([])
    d_outer = nb_dist[:, inner_k: ].ravel()     if inner_k < K else np.array([])

    def robust_geom_median(ds: np.ndarray, fallback: Tuple[float,float]) -> float:
        if ds.size == 0:
            return _geom_mean(*fallback)
        # clip negatives, zeros to tiny positive to keep geom ops valid
        ds = ds[np.isfinite(ds)]
        ds = ds[ds > 1e-9]
        if ds.size == 0:
            return _geom_mean(*fallback)
        lo = float(np.quantile(ds, 0.20))
        hi = float(np.quantile(ds, 0.80))
        lo = max(lo, 1e-6); hi = max(hi, lo + 1e-6)
        return _geom_mean(lo, hi)

    # Fallback bounds for robustness
    b1 = (float(CELL_EDGE_KM_EST), float(R1_MAX_KM))
    b2 = (float(R2_MIN_KM),        float(R2_MAX_KM))

    m1 = robust_geom_median(d_inner, b1)
    m2 = robust_geom_median(d_outer, b2)
    return m1, m2

# =========================
# Linear algebra
# =========================
def power_stationary(P: np.ndarray, tol=1e-12, maxit=10000) -> np.ndarray:
    N = P.shape[0]
    x = np.full(N, 1.0/N, dtype=float)
    for _ in range(maxit):
        x_new = P @ x
        s = x_new.sum()
        x_new = x_new / s if s > 0 else np.full(N, 1.0/N, dtype=float)
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new
        x = x_new
    return x

# =========================
# Transitions persistence (pattern + manifest)
# =========================
def transitions_paths() -> Tuple[str,str]:
    npy = os.path.join(TRANS_DIR, f"pep_P_daily_pattern_m{TIME_RES_MIN}.npy")
    man = os.path.join(TRANS_DIR, f"pep_P_daily_pattern_m{TIME_RES_MIN}.manifest.json")
    return npy, man

def write_transitions(P_daily: List[np.ndarray], nodes: List[str], manifest_path: str, npy_path: str,
                      extra_meta: Optional[dict] = None) -> None:
    if not SAVE_TRANSITIONS:
        return
    arr = np.stack(P_daily, axis=0)
    np.save(npy_path, arr)
    meta = {
        "nodes": nodes,
        "bins_per_day": int(arr.shape[0]),
        "alpha": float(ALPHA),
        "beta": float(BETA),
        "use_exp_decay": bool(USE_EXP_DECAY),
        "dist_scale_km": None if not USE_EXP_DECAY else float(DIST_SCALE_KM),
        "hard_cutoff_km": None if HARD_CUTOFF_KM is None else float(HARD_CUTOFF_KM),
        # schedule & radial params in manifest
        "am_hours_set": sorted(list(AM_HOURS_SET)),
        "midday_hours_set": sorted(list(MIDDAY_HOURS_SET)),
        "pm_hours_set": sorted(list(PM_HOURS_SET)),
        "lambda_dest": float(LAMBDA_DEST),
        "lambda_stay": float(LAMBDA_STAY),
        "r_center_sat_km": float(R_CENTER_SAT_KM),
        "r_periph_sat_km": float(R_PERIPH_SAT_KM),
        "neighbor_scheme": NEIGHBOR_SCHEME,
        "neighbor_k": int(NEIGHBOR_K),
        "neighbor_inner_k": int(NEIGHBOR_INNER_K),
        "neighbor_boost": float(NEIGHBOR_BOOST),
        "neighbor_max_rings": int(NEIGHBOR_MAX_RINGS),
        "time_res_min": int(TIME_RES_MIN),
        "country": COUNTRY,
        "region_name": REGION_NAME,
        "region_bbox": REGION_BBOXES.get(REGION_NAME),
        "region_endpoint_rule": REGION_ENDPOINT_RULE,
        "region_membership_mode": REGION_MEMBERSHIP_MODE,
        "region_buffer_km": float(BUFFER_KM_BY_COUNTRY.get(COUNTRY, 0.0)),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =========================
# Initial masses & initial state
# =========================
def _template_inflow_vector(dfw: pd.DataFrame, nodes: List[str], mode: str,
                            hour_set: Optional[set[int]] = None) -> np.ndarray:
    if COUNT_COL not in dfw.columns or TIME_COL not in dfw.columns:
        return np.ones(len(nodes), dtype=float)
    t = pd.to_datetime(dfw[TIME_COL], errors="coerce", utc=False)
    sel = dfw.copy()
    sel["_t0"] = t
    if mode == "day0":
        d0 = t.dt.floor('D').min()
        sel = sel[t.dt.floor('D') == d0]
    if mode == "window" and hour_set:
        sel = sel[sel["_t0"].dt.hour.isin(list(hour_set))]
    s = sel.groupby(END_COL)[COUNT_COL].sum()
    v = np.array([float(s.get(g, 0.0)) for g in nodes], dtype=float)
    v = np.where(v <= 0, 1.0, v)
    return v


def radial_profile_multiplier_vector(
    nodes: List[str],
    rkm_map: Dict[str, float],
    rc_km: float,
    rp_km: float,
    center_mult: float,
    periph_mult: float
) -> np.ndarray:
    """
    Build per-node multipliers: flat=center_mult for r<=rc, flat=periph_mult for r>=rp,
    linearly interpolated in between.
    """
    rc = float(rc_km); rp = float(rp_km)
    if rp <= rc:  # safety
        rp = rc + 1e-6
    m = []
    for g in nodes:
        r = float(rkm_map[g])
        if r <= rc:
            m.append(center_mult)
        elif r >= rp:
            m.append(periph_mult)
        else:
            u = (r - rc) / (rp - rc)  # 0..1
            m.append(center_mult * (1.0 - u) + periph_mult * u)
    return np.array(m, dtype=float)


def build_initial_masses(dfw: pd.DataFrame, nodes: List[str], rkm_map) -> np.ndarray:
    mode = str(MASS_INIT_MODE).lower()
    if mode == "flat":
        M = np.ones(len(nodes), dtype=float)
    elif mode == "template_inflow_day0":
        M = _template_inflow_vector(dfw, nodes, mode="day0")
    else:  # "template_window_mean"
        hs = set(MASS_INIT_HOUR_SET) if MASS_INIT_HOUR_SET else None
        M = _template_inflow_vector(dfw, nodes, mode="window", hour_set=hs)
    M = M / (M.mean() + 1e-12)


        # Optional: apply radial profile to initial masses (multiplicative), then renormalize mean
    if RADIAL_INIT_ENABLE_MASS:
        mult = radial_profile_multiplier_vector(
            nodes=nodes,
            rkm_map=rkm_map,  # NOTE: rkm_map must be available (see integration below)
            rc_km=RADIAL_INIT_INNER_KM,
            rp_km=RADIAL_INIT_OUTER_KM,
            center_mult=RADIAL_INIT_CENTER_MULT,
            periph_mult=RADIAL_INIT_PERIPH_MULT
        )
        M = M * np.maximum(mult, 1e-12)
        if RADIAL_INIT_NORMALIZE == "mean1":
            M = M / (M.mean() + 1e-12)

    M = np.clip(M, 1e-9, None)
    return M

def periodic_fixed_point(P_daily: List[np.ndarray], tol: float = 1e-12, max_days: int = 2000) -> np.ndarray:
    if not P_daily:
        raise ValueError("P_daily empty")
    N = P_daily[0].shape[0]
    x = np.full(N, 1.0/N, dtype=float)
    for _ in range(max_days):
        x_prev = x.copy()
        for Pb in P_daily:
            x = Pb @ x
            s = x.sum()
            x = x / s if s > 0 else np.full(N, 1.0/N, dtype=float)
        if np.linalg.norm(x - x_prev, 1) < tol:
            return x
    return x

def compute_initial_state(P_daily: List[np.ndarray], nodes: List[str], dfw: pd.DataFrame, rkm_map=None) -> np.ndarray:
    mode = str(INIT_X_MODE).lower()
    N = len(nodes)
    if mode == "flat":
        return np.full(N, 1.0/N, dtype=float)
    if mode == "template_inflow_day0":
        v = _template_inflow_vector(dfw, nodes, mode="day0")
        v_sum = v.sum();  return (v / v_sum) if v_sum > 0 else np.full(N, 1.0/N)
    if mode == "template_window_mean":
        hs = set(MASS_INIT_HOUR_SET) if MASS_INIT_HOUR_SET else None
        v = _template_inflow_vector(dfw, nodes, mode="window", hour_set=hs)
        v_sum = v.sum();  return (v / v_sum) if v_sum > 0 else np.full(N, 1.0/N)
    if mode == "stationary_meanp":
        Pm = np.mean(np.stack(P_daily, axis=0), axis=0)
        x0 = power_stationary(Pm)
        return x0 / (x0.sum() + 1e-12)
    if mode == "periodic_fixed_point":
        x0 = periodic_fixed_point(P_daily)
        return x0 / (x0.sum() + 1e-12)
    Pm = np.mean(np.stack(P_daily, axis=0), axis=0)
    x0 = power_stationary(Pm)

    # Optional: apply radial profile to initial x (multiplicative) then renormalize sum to 1
    if RADIAL_INIT_ENABLE_X:
        # We need rkm_map here; either pass it in or read a cached global.
        # Using cached global to avoid changing this signature:
        # try:
        #     rkm = _RKM_MAP_CACHE  # set in generate()
        # except NameError:
        #     rkm = {g: 0.0 for g in nodes}  # fallback: neutral
        mult = radial_profile_multiplier_vector(
            nodes=nodes,
            rkm_map=rkm_map,
            rc_km=RADIAL_INIT_INNER_KM,
            rp_km=RADIAL_INIT_OUTER_KM,
            center_mult=RADIAL_INIT_CENTER_MULT,
            periph_mult=RADIAL_INIT_PERIPH_MULT
        )
        x0 = x0 * np.maximum(mult, 1e-12)
        s = x0.sum()
        x0 = x0 / (s + 1e-12)


    return x0 / (x0.sum() + 1e-12)

# =========================
# Diagnostics writers (BIN-BASED)
# =========================
def _bin_index_from_timestamp(ts: pd.Timestamp) -> int:
    return int(((ts.hour*60 + ts.minute) - WINDOW_START_HH*60) // TIME_RES_MIN)

def write_pep_temporal_diagnostics(od: pd.DataFrame, centers: set, out_path: str) -> None:
    tmp = od.copy()
    if "_t0" not in tmp.columns:
        tmp["_t0"] = pd.to_datetime(tmp[TIME_COL], errors="coerce", utc=False)
    tmp["_day"]      = tmp["_t0"].dt.floor("D")
    tmp["_bin_idx"]  = tmp["_t0"].apply(_bin_index_from_timestamp).astype(int)
    tmp["_bin_lab"]  = tmp["_t0"].dt.strftime("%H:%M")

    START, END = START_COL, END_COL
    is_center_dest = tmp[END].isin(centers)
    is_center_orig = tmp[START].isin(centers)
    is_self        = tmp[START] == tmp[END]

    keys = ["_day", "_bin_idx", "_bin_lab"]

    inflow_center  = tmp.loc[~is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
    outflow_center = tmp.loc[ is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()
    self_center    = tmp.loc[ is_self        &  is_center_dest].groupby(keys)[COUNT_COL].sum()
    total_bin      = tmp.groupby(keys)[COUNT_COL].sum()

    out = pd.DataFrame({
        "inflow_center": inflow_center,
        "outflow_center": outflow_center,
        "self_center": self_center,
        "total_bin_flow": total_bin
    }).fillna(0).reset_index()

    out["net_out_center"] = out["outflow_center"] - out["inflow_center"]
    out.sort_values(["_day", "_bin_idx"], inplace=True)
    out.to_csv(out_path, index=False)

def write_pep_mean_bin_balance(od: pd.DataFrame, centers: set, out_path: str) -> None:
    tmp_path = out_path + ".__tmp_temporal.csv"
    write_pep_temporal_diagnostics(od, centers, tmp_path)
    df = pd.read_csv(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    grp = df.groupby(["_bin_idx","_bin_lab"], as_index=False)[
        ["inflow_center","outflow_center","self_center","total_bin_flow","net_out_center"]
    ].mean()
    grp.sort_values("_bin_idx", inplace=True)
    grp.to_csv(out_path, index=False)

def write_pep_daily_balance(od: pd.DataFrame, centers: set, out_path: str) -> None:
    tmp_path = out_path + ".__tmp_temporal.csv"
    write_pep_temporal_diagnostics(od, centers, tmp_path)
    df = pd.read_csv(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    daily = (
        df.groupby("_day", as_index=False)[
            ["inflow_center","outflow_center","self_center","total_bin_flow","net_out_center"]
        ].sum()
    )
    daily.rename(columns={"total_bin_flow":"total_day_flow"}, inplace=True)
    daily.sort_values("_day", inplace=True)
    daily.to_csv(out_path, index=False)

# =========================
# VALIDATION HELPERS
# =========================
def _read_generated_od() -> pd.DataFrame:
    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y%m%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y%m%d")
    tag = f"{start_tag}_{end_tag}_m{TIME_RES_MIN}"
    path = os.path.join(OUTPUT_DIR, f"pep_od_{tag}.csv")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    if not os.path.exists(path):
        raise FileNotFoundError("No aggregated OD found. Run in generate mode first or enable WRITE_OD_AGGREGATE.")
    df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    if "_t0" not in df.columns:
        if TIME_COL not in df.columns:
            raise ValueError("Aggregated OD missing time column.")
        df["_t0"] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=False)
    return df

def _load_transitions() -> Tuple[List[str], np.ndarray, dict]:
    npy, man = transitions_paths()
    if not os.path.exists(npy) or not os.path.exists(man):
        raise FileNotFoundError("Saved transitions not found (enable SAVE_TRANSITIONS).")
    P_daily = np.load(npy)
    with open(man, "r", encoding="utf-8") as f:
        meta = json.load(f)
    nodes = meta["nodes"]
    return nodes, P_daily, meta

def rebuild_empirical_P_for_bin(od_bin: pd.DataFrame, nodes: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {g:i for i,g in enumerate(nodes)}
    N = len(nodes)
    flows = np.zeros((N, N), dtype=float)
    for _, row in od_bin.iterrows():
        sj = row[START_COL]; di = row[END_COL]; c = float(row.get(COUNT_COL, 0.0))
        if sj in idx and di in idx:
            flows[idx[di], idx[sj]] += c
    col_sums = flows.sum(axis=0)
    P_emp = np.zeros_like(flows)
    mask = col_sums > 0
    P_emp[:, mask] = flows[:, mask] / col_sums[mask]
    return P_emp, col_sums, mask

def kl_l1_columns(P_emp: np.ndarray, P_mod: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    used = np.where(mask)[0]
    if used.size == 0:
        return np.array([]), np.array([]), used
    Pe = P_emp[:, used] + eps
    Pm = P_mod[:, used] + eps
    Pe /= Pe.sum(axis=0, keepdims=True)
    Pm /= Pm.sum(axis=0, keepdims=True)
    kl = np.sum(Pe * (np.log(Pe) - np.log(Pm)), axis=0)
    l1 = np.sum(np.abs(Pe - Pm), axis=0)
    return kl, l1, used

def _final_population_series(agent_state: np.ndarray, N: int) -> np.ndarray:
    """Return counts per node index from the final agent_state."""
    return np.bincount(agent_state, minlength=N).astype(int)

def write_final_population(
    agent_state: np.ndarray,
    nodes: List[str],
    centers: List[str],
    rkm_map: Dict[str, float],
    out_dir: str,
    tag: str,
    pool_size: int,
) -> None:
    """Save final population by tile and aggregates (center/non-center and radial bands)."""
    N = len(nodes)
    counts = _final_population_series(agent_state, N)
    shares = counts / float(max(1, pool_size))

    centers_set = set(centers)
    is_center = np.array([g in centers_set for g in nodes], dtype=bool)
    r_km = np.array([float(rkm_map[g]) for g in nodes], dtype=float)

    # Per-tile table
    df_tiles = pd.DataFrame({
        "geohash5": nodes,
        "final_count": counts,
        "final_share": shares,
        "r_km": r_km,
        "is_center": is_center,
        "is_periphery_r_ge_rp": r_km >= float(R_PERIPH_SAT_KM),
        "is_inner_r_lt_rc": r_km < float(R_CENTER_SAT_KM),
    }).sort_values(["is_center","final_count"], ascending=[False, False])

    path_tiles = os.path.join(out_dir, f"pep_final_population_{tag}.csv")
    df_tiles.to_csv(path_tiles, index=False)

    # Aggregates
    def _sum(mask: np.ndarray) -> Tuple[int, float]:
        c = int(counts[mask].sum())
        return c, c / float(max(1, pool_size))

    agg_rows = []
    c_cnt, c_sh = _sum(is_center)
    nc_cnt, nc_sh = _sum(~is_center)
    inner_cnt, inner_sh = _sum(r_km < float(R_CENTER_SAT_KM))
    mid_cnt, mid_sh = _sum((r_km >= float(R_CENTER_SAT_KM)) & (r_km < float(R_PERIPH_SAT_KM)))
    peri_cnt, peri_sh = _sum(r_km >= float(R_PERIPH_SAT_KM))

    agg_rows += [
        {"group": "center_set", "count": c_cnt, "share": c_sh},
        {"group": "non_center", "count": nc_cnt, "share": nc_sh},
        {"group": f"inner_r<rc (rc={R_CENTER_SAT_KM}km)", "count": inner_cnt, "share": inner_sh},
        {"group": f"mid_rc<=r<rp (rp={R_PERIPH_SAT_KM}km)", "count": mid_cnt, "share": mid_sh},
        {"group": f"periphery_r>=rp", "count": peri_cnt, "share": peri_sh},
        {"group": "TOTAL", "count": int(counts.sum()), "share": float(counts.sum()/float(max(1, pool_size)))},
    ]
    df_agg = pd.DataFrame(agg_rows)
    path_agg = os.path.join(out_dir, f"pep_final_population_aggregates_{tag}.csv")
    df_agg.to_csv(path_agg, index=False)

    info(f"[WRITE] final population by tile: {path_tiles}")
    info(f"[WRITE] final population aggregates: {path_agg}")

# =========================
# validate()
# =========================
def validate():
    banner("PEP VALIDATE")

    od = _read_generated_od()
    od = od.loc[
        (od["_t0"].dt.floor("D") >= pd.Timestamp(SWEEP_START_DATE)) &
        (od["_t0"].dt.floor("D") <= pd.Timestamp(SWEEP_END_DATE)) &
        (od["_t0"].dt.hour >= WINDOW_START_HH) & (od["_t0"].dt.hour < WINDOW_END_HH)
    ].copy()

    nodes, P_daily, meta = _load_transitions()
    centers = set(meta.get("centers", []))
    bins_per_day = int(meta.get("bins_per_day", max(1, int(((WINDOW_END_HH-WINDOW_START_HH)*60)//TIME_RES_MIN))))

    # Empirical vs model comparison per exact bin
    comp_rows = []
    bins = sorted(od["_t0"].unique())
    for t in bins:
        b = ((t.hour*60 + t.minute) - WINDOW_START_HH*60) // TIME_RES_MIN
        if b < 0 or b >= bins_per_day:
            continue
        P_mod = P_daily[int(b)]
        P_emp_m, col_mass_used, col_mask = rebuild_empirical_P_for_bin(od[od["_t0"] == t], nodes)
        if col_mass_used.size == 0 or not np.any(col_mask):
            comp_rows.append({
                "bin_time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "bin_index": int(b),
                "kl_emp_vs_model": float("nan"),
                "l1_emp_vs_model": float("nan"),
                "emp_total": int(od.loc[od["_t0"] == t, COUNT_COL].sum()),
                "used_columns": 0,
                "total_origins": int(len(nodes)),
            })
            continue
        kl_cols, l1_cols, used_cols = kl_l1_columns(P_emp_m, P_mod, col_mask)
        w = col_mass_used[col_mask]
        w = w / (w.sum() + 1e-12)
        kl_w = float(np.sum(kl_cols * w))
        l1_w = float(np.sum(l1_cols * w))
        comp_rows.append({
            "bin_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "bin_index": int(b),
            "kl_emp_vs_model": kl_w,
            "l1_emp_vs_model": l1_w,
            "emp_total": int(col_mass_used.sum()),
            "used_columns": int(used_cols.size),
            "total_origins": int(len(nodes)),
        })

    # comp_df = pd.DataFrame(comp_rows)
    # comp_path = os.path.join(OUTPUT_DIR, "pep_model_empirical_compare.csv")
    # comp_df.to_csv(comp_path, index=False)
    # info(f"[WRITE] model vs empirical (per-bin, mass-weighted): {comp_path}")

    # Diagnostics: temporal/day-bin/daily balances
    # Empirical vs model comparison per exact bin
    if ENABLE_DIAG_MODEL_EMP_COMPARE:
        comp_df = pd.DataFrame(comp_rows)
        comp_path = os.path.join(OUTPUT_DIR, "pep_model_empirical_compare.csv")
        comp_df.to_csv(comp_path, index=False)
        info(f"[WRITE] model vs empirical (per-bin, mass-weighted): {comp_path}")
    else:
        info("[SKIP] model vs empirical compare (disabled)")

    # Diagnostics: temporal/day-bin/daily balances
    if ENABLE_DIAG_TEMPORAL:
        td_path = os.path.join(OUTPUT_DIR, "pep_temporal_diagnostics.csv")
        write_pep_temporal_diagnostics(od, centers, td_path)
        info(f"[WRITE] temporal diagnostics: {td_path}")
    else:
        info("[SKIP] temporal diagnostics (disabled)")

    if ENABLE_DIAG_MEAN_BIN:
        mb_path = os.path.join(OUTPUT_DIR, "pep_mean_bin_balance.csv")
        write_pep_mean_bin_balance(od, centers, mb_path)
        info(f"[WRITE] mean bin balance: {mb_path}")
    else:
        info("[SKIP] mean bin balance (disabled)")

    if ENABLE_DIAG_DAILY:
        dl_path = os.path.join(OUTPUT_DIR, "pep_daily_balance.csv")
        write_pep_daily_balance(od, centers, dl_path)
        info(f"[WRITE] daily balance: {dl_path}")
    else:
        info("[SKIP] daily balance (disabled)")


    # Periodicity (sanity): end-of-day vs start-of-day L1 on destination totals
    od["_day"] = od["_t0"].dt.floor("D")
    l1_vals = []
    for day, group in od.groupby("_day"):
        times = sorted(group["_t0"].unique())
        if len(times) < 2:
            continue
        t_first, t_last = times[0], times[-1]
        g_first = group[group["_t0"] == t_first]
        g_last  = group[group["_t0"] == t_last]
        dst_first = g_first.groupby(END_COL)[COUNT_COL].sum()
        dst_last  = g_last .groupby(END_COL)[COUNT_COL].sum()
        idx = sorted(set(dst_first.index).union(dst_last.index))
        a = np.array([float(dst_first.get(k, 0.0)) for k in idx], dtype=float)
        b = np.array([float(dst_last .get(k, 0.0)) for k in idx], dtype=float)
        l1 = float(np.sum(np.abs(a - b)))
        l1_vals.append(l1)
    if l1_vals:
        info(f"[PERIODIC] bins/day≈{bins_per_day}, mean L1(end_of_day, start) = {np.mean(l1_vals):.3f}")
    else:
        warn("[PERIODIC] insufficient bins per day to compute L1.")

    info("Validation complete.")

# =========================
# Core: generate (EVERY agent transitions & emits)
# =========================
def generate():
    banner("PEP GENERATION")
    rng = np.random.default_rng(SEED)

    # Time bins for one day (pattern)
    bins_per_day = int(((WINDOW_END_HH - WINDOW_START_HH) * 60) // TIME_RES_MIN)
    minutes_in_day = [WINDOW_START_HH*60 + b*TIME_RES_MIN for b in range(bins_per_day)]
    # Diagnostics collector
    neighbor_diag_rows = []  # (only used if ENABLE_NEIGHBOR_DIAGNOSTICS)

    # Template & region filter
    if USE_TEMPLATE and os.path.exists(TEMPLATE_PATH):
        tpl = pd.read_csv(TEMPLATE_PATH)
        dfw = window_slice_template(tpl := tpl)
        dfw = apply_region_filter(dfw, REGION_NAME)
        nodes = sorted(set(dfw[START_COL]).union(dfw[END_COL]))
        if len(nodes) == 0:
            raise ValueError("No nodes discovered from template after region filter.")
        info(f"NODES (unique geohash{GEOHASH_PRECISION}): {len(nodes)}")

        centers = infer_centers_by_inflow(dfw, CENTER_K)
        info(f"Center tiles (K={CENTER_K}): {centers}")
    else:
        raise FileNotFoundError("Template required (USE_TEMPLATE=True).")

    # Radial distances (km) and optional diagnostic rings
    center_latlon = centroid_of_tiles(centers)
    rkm_map = radial_km(nodes, center_latlon)
    _ = diagnostic_ring_labels(nodes, rkm_map, DIAGNOSTIC_RING_BINS)

    # --- Auto-tune saturation radii (optional) ---
    if AUTO_TUNE_SAT_RADII:
        rvals = np.array([rkm_map[g] for g in nodes], dtype=float)
        # periphery at percentile
        rp_auto = np.quantile(rvals, float(AUTO_PERIPH_Q))
        # center = max(farthest center + pad, small)
        far_center = max(rkm_map.get(g, 0.0) for g in centers) if centers else 0.0
        rc_auto = max(float(far_center) + float(AUTO_CENTER_PAD_KM), 0.5 * float(CELL_EDGE_KM_EST))

        # apply mode
        if AUTO_TUNE_MODE == "replace":
            new_rc = rc_auto
            new_rp = rp_auto
        else:  # "floor": only raise, never shrink
            new_rc = max(float(R_CENTER_SAT_KM), rc_auto)
            new_rp = max(float(R_PERIPH_SAT_KM), rp_auto)

        # allow hard overrides
        if RC_OVERRIDE_KM is not None: new_rc = float(RC_OVERRIDE_KM)
        if RP_OVERRIDE_KM is not None: new_rp = float(RP_OVERRIDE_KM)

        globals()["R_CENTER_SAT_KM"] = float(new_rc)
        globals()["R_PERIPH_SAT_KM"] = float(new_rp)
        info(f"[AUTO-R] R_CENTER_SAT_KM={R_CENTER_SAT_KM:.2f} km, R_PERIPH_SAT_KM={R_PERIPH_SAT_KM:.2f} km")

    nb_idx, nb_dist = neighbor_indices(nodes, NEIGHBOR_K, NEIGHBOR_SCHEME, NEIGHBOR_MAX_RINGS)

    # Optional: precompute pairwise envelopes for lattice mode
    pair_env_dmin = pair_env_dmax = None
    if ENABLE_PAIRWISE_DISTANCE_ENVELOPES:
        info("[PAIR-ENV] Precomputing pair-specific distance envelopes (lattice mode).")
        pair_env_dmin, pair_env_dmax = precompute_pairwise_envelopes_generic(nodes, nb_idx)

        if DUMP_PAIRWISE_ENVELOPES_JSON:
            # Dump a compact JSON with (origin, neighbor, dmin, dmax)
            env_dump = []
            for j in range(len(nodes)):
                for r in range(nb_idx.shape[1]):
                    i = int(nb_idx[j, r])
                    env_dump.append({
                        "origin": nodes[j],
                        "dest": nodes[i],
                        "rank": int(r),
                        "dmin_m": float(pair_env_dmin[j, r] * 1000.0),
                        "dmax_m": float(pair_env_dmax[j, r] * 1000.0),
                    })
            dump_path = os.path.join(OUTPUT_DIR, "pep_pairwise_envelopes.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(env_dump, f)
            info(f"[PAIR-ENV] Wrote envelope dump: {dump_path}")

    # --- Optional: auto-tune hop distance medians (default OFF; purely additive) ---
    if ENABLE_AUTO_DISTANCE_MEDIANS:
        if AUTO_DISTANCE_MEDIAN_STRATEGY == "empirical_neighbors":
            m1, m2 = _auto_distance_medians_from_neighbors(nb_dist, int(NEIGHBOR_INNER_K))
        else:  # "geom_bounds"
            m1, m2 = _auto_distance_medians_from_bounds()

        # Update medians for hop groups 1 and 2; keep group 0 unchanged
        LN_MEDIANS[1] = float(m1)
        LN_MEDIANS[2] = float(m2)

        if AUTO_LN_SIGMA is not None:
            globals()["LN_SIGMA"] = float(AUTO_LN_SIGMA)

        info(f"[AUTO-HOPS] LN_MEDIANS[1]={LN_MEDIANS[1]:.3f} km, LN_MEDIANS[2]={LN_MEDIANS[2]:.3f} km; "
            f"LN_SIGMA={LN_SIGMA:.3f}")

    # Base destination masses
    # M_base = build_initial_masses(dfw, nodes)
    M_base = build_initial_masses(dfw, nodes, rkm_map)

    # Build daily pattern once
    P_daily = [build_P_for_minute(nodes, rkm_map, set(centers), mday, M_base, nb_idx, nb_dist)
               for mday in minutes_in_day]

    # Initial state x0
    x0 = compute_initial_state(P_daily, nodes, dfw,rkm_map=rkm_map)

    # Persist transitions pattern
    trans_npy, manifest = transitions_paths()
    # write_transitions(P_daily, nodes, manifest, trans_npy,
    #                   extra_meta={"centers": centers, "seed": SEED,
    #                               "mass_init_mode": MASS_INIT_MODE,
    #                               "init_x_mode": INIT_X_MODE})
    write_transitions(
        P_daily, nodes, manifest, trans_npy,
        extra_meta={
            "centers": centers,
            "seed": SEED,
            "mass_init_mode": MASS_INIT_MODE,
            "init_x_mode": INIT_X_MODE,
            "enable_auto_distance_medians": ENABLE_AUTO_DISTANCE_MEDIANS,
            "auto_distance_median_strategy": AUTO_DISTANCE_MEDIAN_STRATEGY,
            "auto_ln_sigma": AUTO_LN_SIGMA,
            "ln_medians_effective_km": {k: float(v) for k, v in LN_MEDIANS.items()},
            "ln_sigma_effective": float(LN_SIGMA),
        },
    )


    
    # Build a per-bin totals frame from the template for TOTAL_PER_BIN_MODE
    # This assumes dfw has TIME_COL and (optionally) COUNT_COL.
    template_df = dfw.copy()
    if TIME_COL not in template_df.columns:
        raise RuntimeError("Template is missing TIME_COL; cannot compute per-bin means for TOTAL_PER_BIN_MODE")

    # Parse time and keep only rows inside the configured window (hours)
    template_df["_t0"] = pd.to_datetime(template_df[TIME_COL], errors="coerce", utc=False)
    template_df = template_df[
        (template_df["_t0"].dt.hour >= WINDOW_START_HH) &
        (template_df["_t0"].dt.hour <  WINDOW_END_HH)
    ].copy()

    # Compute bin index for the configured TIME_RES_MIN
    template_df["bin"] = ((template_df["_t0"].dt.hour*60 + template_df["_t0"].dt.minute) - WINDOW_START_HH*60) // TIME_RES_MIN
    template_df["local_date"] = template_df["_t0"].dt.floor("D")

    # Determine trips per row
    if COUNT_COL in template_df.columns:
        template_df["_trips"] = template_df[COUNT_COL].astype(float)
    else:
        template_df["_trips"] = 1.0  # treat each row as one trip if no counts

    # Aggregate total trips per (day, bin)
    template_bin_totals = (
        template_df.groupby(["local_date", "bin"], as_index=False)["_trips"].sum()
        .rename(columns={"_trips": "trip_count"})
    )



    # Agent pool (ALL agents) -- derive POOL_SIZE from TOTAL_PER_BIN_MODE
    N = len(nodes)
    node_to_idx = {g:i for i,g in enumerate(nodes)}

    if TOTAL_PER_BIN_MODE == "fixed":
        total_per_bin = float(TOTAL_PER_BIN_FIXED)
        info(f"[TOTALS] Fixed total per bin: {total_per_bin:.0f}")

    elif TOTAL_PER_BIN_MODE == "template_mean":
        if template_bin_totals.empty:
            raise RuntimeError("[TOTALS] template_bin_totals is empty; cannot compute template_mean")
        # mean across all days and bins of total trips per bin
        total_per_bin = float(template_bin_totals.groupby("bin")["trip_count"].sum().mean())
        info(f"[TOTALS] Using mean total per bin from full template window: {total_per_bin:.0f}")

    elif TOTAL_PER_BIN_MODE == "template_mean_day0":
        if template_bin_totals.empty:
            raise RuntimeError("[TOTALS] template_bin_totals is empty; cannot compute template_mean_day0")
        day0 = template_bin_totals["local_date"].min()
        df0 = template_bin_totals.loc[template_bin_totals["local_date"] == day0]
        if df0.empty:
            warn("[TOTALS] No rows for day0; falling back to full template mean")
            total_per_bin = float(template_bin_totals.groupby("bin")["trip_count"].sum().mean())
        else:
            total_per_bin = float(df0.groupby("bin")["trip_count"].sum().mean())
        info(f"[TOTALS] Using mean total per bin from day0 ({day0.date()}): {total_per_bin:.0f}")

    else:
        raise ValueError(f"Unknown TOTAL_PER_BIN_MODE: {TOTAL_PER_BIN_MODE}")

    POOL_SIZE = max(1, int(round(total_per_bin)))
    agent_ids = np.array([f"P{(k+1):06d}" for k in range(POOL_SIZE)], dtype=object)


    # initial positions from x0
    agent_state = rng.choice(N, size=POOL_SIZE, p=x0, replace=True)

    # rank maps for hop group classification (distance sampling)
    rank_maps = []
    K = nb_idx.shape[1]
    for j in range(N):
        ranks = {int(nb_idx[j, r]): r for r in range(K)}
        rank_maps.append(ranks)

    out_rows: List[Tuple[str,str,str,int,str,float]] = []

    day = pd.to_datetime(SWEEP_START_DATE)
    end_day = pd.to_datetime(SWEEP_END_DATE)

    while day <= end_day:
        for b, mday in enumerate(minutes_in_day):
            tbin = day + pd.Timedelta(minutes=int(mday)) - pd.Timedelta(minutes=WINDOW_START_HH*60)
            P = P_daily[b]
            # inline progress indicator (updates each simulated day)
            print(f"\r[DAY] {day.date()} in progress...", end="", flush=True)

            # --- Neighbor diagnostics: count non-zero destinations per origin/column ---
            if ENABLE_NEIGHBOR_DIAGNOSTICS:
                # For each origin j: how many destinations i have P[i,j] > 0?
                # We'll also split out self vs non-self, and compare to K.
                Ncols = P.shape[1]
                for j in range(Ncols):
                    col = P[:, j]
                    nnz_total = int(np.count_nonzero(col > 0.0))
                    has_self = bool(col[j] > 0.0)
                    nnz_nonself = int(nnz_total - (1 if has_self else 0))

                    # How many neighbors were even eligible (excluding self)?
                    # This uses the neighbor list we constructed; some may be zero after cutoff.
                    neigh_list = nb_idx[j]
                    neigh_excl_self = neigh_list[neigh_list != j]
                    eligible_neighbors = int(neigh_excl_self.size)  # typically K-1

                    # Among eligible neighbors, how many actually got >0 prob?
                    nnz_among_neighbors = int(np.count_nonzero(col[neigh_excl_self] > 0.0))

                    # Stay prob & spread for context
                    pst = float(P[j, j])
                    spread = float(1.0 - pst)

                    # Any entries outside the neighbor set that got prob? (should be 0)
                    outside = np.setdiff1d(np.where(col > 0.0)[0], neigh_list, assume_unique=False)
                    nnz_outside_neighbors = int(outside.size)

                    neighbor_diag_rows.append({
                        "sim_day": int(day.strftime("%Y%m%d")),
                        "bin_index": int(b),
                        "minute_of_day": int(mday),
                        "origin_geohash5": nodes[j],
                        "pst_self": pst,
                        "spread_nonself": spread,
                        "nnz_total": nnz_total,
                        "nnz_nonself": nnz_nonself,
                        "eligible_neighbors": eligible_neighbors,
                        "nnz_among_neighbors": nnz_among_neighbors,
                        "nnz_outside_neighbors": nnz_outside_neighbors,  # should be 0
                        "NEIGHBOR_K_config": int(NEIGHBOR_K),
                        "NEIGHBOR_INNER_K_config": int(NEIGHBOR_INNER_K),
                    })
                    # print("HERE")

            # === Markov step: EVERY agent transitions once ===
            js_all = agent_state
            next_state = agent_state.copy()

            uniq_js_all, inv_all = np.unique(js_all, return_inverse=True)
            for u_idx, j in enumerate(uniq_js_all):
                idx_j = np.where(inv_all == u_idx)[0]
                n_j = idx_j.size
                if n_j == 0:
                    continue
                col = P[:, j].astype(float)
                col = np.clip(col, 0.0, None)
                ssum = float(col.sum())
                if ssum <= 0.0:
                    dest_counts = np.zeros(N, dtype=int); dest_counts[j] = n_j
                else:
                    col /= ssum
                    dest_counts = rng.multinomial(n=n_j, pvals=col)
                assign = np.repeat(np.arange(N), dest_counts)
                rng.shuffle(assign)
                next_state[idx_j] = assign

            # === Emit a row for EVERY agent ===
            for a in range(POOL_SIZE):
                j = int(js_all[a]); i = int(next_state[a])
                if i == j:
                    hop_group = 0
                else:
                    rnk = rank_maps[j].get(i, NEIGHBOR_K + 1)
                    hop_group = 1 if rnk < NEIGHBOR_INNER_K else 2
                # d_km = sample_hop_distance_km(hop_group, rng)

                # Prefer pair-specific uniform envelope in lattice mode if enabled, else fall back
                d_km = None
                if ENABLE_PAIRWISE_DISTANCE_ENVELOPES and NEIGHBOR_SCHEME == "lattice" and pair_env_dmin is not None:
                    d_km = sample_pair_distance_uniform(
                        j=j,
                        i=i,
                        nb_idx_row=nb_idx[j],
                        dmin_row=pair_env_dmin[j] if pair_env_dmin is not None else None,
                        dmax_row=pair_env_dmax[j] if pair_env_dmax is not None else None,
                        rng=rng
                    )
                if d_km is None:
                    # fallback: original hop-group sampler (lognormal-with-truncation, or whatever you switched on)
                    d_km = sample_hop_distance_km(hop_group, rng)


                out_rows.append((
                    agent_ids[a],                  # PEP_ID
                    nodes[js_all[a]],              # start_geohash5
                    nodes[i],                      # end_geohash5
                    int(day.strftime("%Y%m%d")),   # local_date
                    (tbin.strftime("%Y-%m-%d %H:%M:%S")),  # local_time
                    float(d_km*1000.0),            # length_m
                ))

            agent_state = next_state
        day += pd.Timedelta(days=1)

    # ======================
    # Write outputs
    # ======================
    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y%m%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y%m%d")
    tag = f"{start_tag}_{end_tag}_m{TIME_RES_MIN}"

    raw_cols = ["PEP_ID", START_COL, END_COL, DATE_COL, TIME_COL, "length_m"]
    pep_df = pd.DataFrame(out_rows, columns=raw_cols)
    # Stable ordering for raw trip file
    pep_df.sort_values(
        by=[DATE_COL, TIME_COL, START_COL, END_COL],
        inplace=True,
        ignore_index=True
    )

    if WRITE_RAW:
        raw_path = os.path.join(OUTPUT_DIR, f"pep_raw_{tag}.csv")
        if COMPRESS_OUTPUTS:
            raw_path += ".gz"
            with gzip.open(raw_path, "wt", encoding="utf-8") as f:
                pep_df.to_csv(f, index=False)
        else:
            pep_df.to_csv(raw_path, index=False)
        info(f"Wrote raw PEP rows: {raw_path}")

    # Optional: stable ordering for OD aggregates

    if WRITE_OD_AGGREGATE:
        g = pep_df.groupby([START_COL, END_COL, DATE_COL, TIME_COL], as_index=False)
        od = g.agg(trip_count=("length_m", "size"),
                   m_length_m=("length_m", "mean"),
                   mdn_length_m=("length_m", "median"))
        od.sort_values(
            by=[DATE_COL, TIME_COL, START_COL, END_COL],
            inplace=True,
            ignore_index=True
        )
        od_path = os.path.join(OUTPUT_DIR, f"pep_od_{tag}.csv")
        if COMPRESS_OUTPUTS:
            od_path += ".gz"
            with gzip.open(od_path, "wt", encoding="utf-8") as f:
                od.to_csv(f, index=False)
        else:
            od.to_csv(od_path, index=False)
        # od.to_csv(od_path, index=False)
        info(f"Wrote OD aggregate: {od_path}")

    if ENABLE_NEIGHBOR_DIAGNOSTICS and neighbor_diag_rows:
        diag_df = pd.DataFrame(neighbor_diag_rows)
        diag_path = os.path.join(OUTPUT_DIR, f"pep_neighbor_diagnostics_{tag}.csv")
        diag_df.to_csv(diag_path, index=False)
        info(f"[WRITE] neighbor diagnostics (per-bin, per-origin): {diag_path}")

        # A compact summary per origin (across the whole run)
        summary = diag_df.groupby("origin_geohash5", as_index=False).agg(
            bins=("bin_index", "size"),
            mean_nnz_nonself=("nnz_nonself", "mean"),
            max_nnz_nonself=("nnz_nonself", "max"),
            mean_eligible_neighbors=("eligible_neighbors", "mean"),
            max_eligible_neighbors=("eligible_neighbors", "max"),
            any_outside_neighbors=("nnz_outside_neighbors", lambda s: int((s > 0).any())),
        )
        sum_path = os.path.join(OUTPUT_DIR, f"pep_neighbor_diagnostics_summary_{tag}.csv")
        summary.to_csv(sum_path, index=False)
        info(f"[WRITE] neighbor diagnostics summary: {sum_path}")

    # --- NEW: write final population vector and aggregates ---
    write_final_population(
        agent_state=agent_state,
        nodes=nodes,
        centers=centers,
        rkm_map=rkm_map,
        out_dir=OUTPUT_DIR,
        tag=tag,
        pool_size=POOL_SIZE,
    )

    # --- Write neighbor diagnostics (if enabled) ---


    if VALIDATE_ON_GENERATE:
        try:
            validate()
        except Exception as e:
            warn(f"Validation after generate failed: {e}")

    info("Generation complete.")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    if MODE == "generate":
        generate()
    elif MODE == "validate":
        validate()
    else:
        raise ValueError(f"Unknown MODE={MODE}")
