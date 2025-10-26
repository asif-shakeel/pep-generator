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
BASE_DIR   = "/Users/asif/Documents/nm24" #os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
TRANS_DIR  = os.path.join(OUTPUT_DIR, "transitions")
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
SWEEP_END_DATE   = "2019-07-10"   # inclusive
WINDOW_START_HH  = 0               # inclusive
WINDOW_END_HH    = 24              # exclusive
TIME_RES_MIN     = 45              # bin size (minutes)

# =========================
# Files / plots / randomness (RESTORED)
# =========================
WRITE_GZIP             = False #  True  #   # legacy name; maps to COMPRESS_OUTPUTS
WRITE_RAW              = True # False
WRITE_OD_AGGREGATE     = True
SAVE_TRANSITIONS       = True
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
# Gravity & distance model
# =========================
ALPHA          = 0.50            # emphasize destination mass
BETA           = 1.8             # gentler decay → reach the center from far
USE_EXP_DECAY  = False
DIST_SCALE_KM  = 5.0
HARD_CUTOFF_KM = None

# Derived
GAMMA_DECAY     = 0.0
DIST_DECAY_MODE = "power"

# =========================
# Center & neighbors
# =========================
CENTER_K = 25                    # broad “center set” for diagnostics
NEIGHBOR_SCHEME     = "lattice" # "knn"
NEIGHBOR_K          = 49         # large fan-out so periphery can reach inward
NEIGHBOR_INNER_K    = 25
NEIGHBOR_BOOST      = 4.0        # mild local cohesion; center bias comes from masses/λ
NEIGHBOR_MAX_RINGS  = 4
NEIGHBOR_MAX_RINGS = 8   # or higher (and maybe NEIGHBOR_K accordingly)
# NEIGHBOR_K = (2*NEIGHBOR_MAX_RINGS+1)**2  # or just something large like 225–400
# NEIGHBOR_INNER_K    = NEIGHBOR_K
ENABLE_NEIGHBOR_DIAGNOSTICS = False
# =========================
# Center selection config
# =========================
# Which metric defines the "center set"
#   - "rank_by_radius"  : pick the K tiles with smallest radius (km) from a reference center
#   - "radius_threshold": pick all tiles with r_km <= CENTER_RADIUS_KM
#   - "inflow_topk"     : pick the K tiles with largest template inflow (END tile counts)
CENTER_SELECTION_MODE = "rank_by_radius"   # {"rank_by_radius","radius_threshold","inflow_topk"}


# For radius_threshold mode only
CENTER_RADIUS_KM = 10.0

# Where the geometric radius is measured from (used by rank_by_radius / radius_threshold)
#   - "inflow_centroid": centroid of inflow Top-K (robust if template exists)
#   - "all_nodes_centroid": centroid of all tiles (purely geometric)
#   - "custom": use CENTER_REF_LATLON explicitly
CENTER_REFERENCE = "all_nodes_centroid"   # {"inflow_centroid","all_nodes_centroid","custom"}
CENTER_REF_LATLON = (None, None)       # e.g., (19.4326, -99.1332)  # (lat, lon) if CENTER_REFERENCE == "custom"

# =========================
# Distance sampling
# =========================
ENABLE_PAIRWISE_DISTANCE_ENVELOPES = False
ENABLE_AUTO_DISTANCE_MEDIANS = False
AUTO_DISTANCE_MEDIAN_STRATEGY = "geom_bounds"
AUTO_LN_SIGMA = None
ENABLE_UNIFORM_DISTANCE_SAMPLING = True
SELF_DISTANCE_MODE = "single_diag"
DUMP_PAIRWISE_ENVELOPES_JSON = False

# =========================
# Diagnostics toggles
# =========================
ENABLE_DIAG_MODEL_EMP_COMPARE = False
ENABLE_DIAG_TEMPORAL          = True
ENABLE_DIAG_MEAN_BIN          = True
ENABLE_DIAG_DAILY             = True

# =========================
# Initial mass & initial state
# =========================
# Destination masses: center-heavy all day; start state slightly periphery-biased.
MASS_INIT_MODE     = "center_periph"  # "flat" # 
MASS_INIT_HOUR_SET = None
INIT_X_MODE        = "center_periph"  # "flat" # 


DETERMINISTIC_INITIAL_ASSIGNMENT = True  # new global toggle

# (center_value, periphery_value)
MASS_CENTER_PERIPH = (20.0, 1)    # center tiles 2× heavier, periphery 0.8×
X_CENTER_PERIPH    = (1, 10)    # fewer people start in center, more outside

# Radial reshaping (on for both masses and x)
RADIAL_INIT_ENABLE_MASS = False
RADIAL_INIT_ENABLE_X    = False

# km plateaus that roughly bracket the metro footprint
RADIAL_INIT_INNER_KM = 10.0
RADIAL_INIT_OUTER_KM = 50.0

# Heavier center for destination masses; lighter center for initial x (more people start outside)
RADIAL_INIT_CENTER_MULT  = 2.0   # MASS multipliers → center ≈2× attractive
RADIAL_INIT_PERIPH_MULT  = 0.6
RADIAL_INIT_NORMALIZE    = "mean1"



# =========================
# Per-bin population control
# =========================
TOTAL_PER_BIN_MODE  = "fixed"
TOTAL_PER_BIN_FIXED = 12000

# =========================
# Saturation radii (fixed for determinism)
# =========================
AUTO_TUNE_SAT_RADII = False
AUTO_TUNE_MODE      = "floor"
AUTO_PERIPH_Q       = 0.92
RC_OVERRIDE_KM      = None
RP_OVERRIDE_KM      = None

# Strong radial gradient used by g(r)
R_CENTER_SAT_KM  = 8.0            # small center plateau → high g(r) near core
R_PERIPH_SAT_KM  = 55.0           # large periphery plateau

# =========================
# Temporal phases (AM in, MIDDAY mild in, PM out, NIGHT no move)
# =========================
AM_HOURS_SET     = {6, 7, 8, 9, 10}                  # strong inward
MIDDAY_HOURS_SET = {11, 12, 13}                      # moderate inward
PM_HOURS_SET     = {14, 15, 16, 17, 18, 19, 20, 21}  # strong outward

# Phase weights (used by phase_s_of_minute)
PHASE_VALUE_AM      = +1.0   # strong inward
PHASE_VALUE_MIDDAY  = +0.4   # moderate inward
PHASE_VALUE_PM      = -1.0   # strong outward
PHASE_VALUE_ELSE    =  0.0   # night: neutral (combine with high stay below → ~no movement)

# =========================
# Stays
# =========================
# Night "no movement": set a high baseline; λ will drop/raise pst in active hours.
P_STAY_BASE = 0.8          # ≈2% spread at night (s=0) → near standstill
P_STAY_MIN  = 0.02            # safety rails
P_STAY_MAX  = 0.98


# =========================
# Stays (region-specific)
# =========================
# Baseline stickiness (center, periphery)
P_STAY_BASE_MODE = (0.05, 0.95)  # tuple (center, periphery)
ENABLE_CENTER_PERIPH_STAY = True

# --- NEW: time-varying baseline schedules (piecewise linear, repeat daily) ---
ENABLE_STAY_BASE_SCHEDULE = True  # master toggle


# Runtime arrays (filled in generate())
_CENTER_BASELINE_BY_BIN: np.ndarray | None = None
_PERIPH_BASELINE_BY_BIN: np.ndarray | None = None

CENTER_STAY_SCHEDULE = [
    ("06:00", "08:00", 0.80),  # 0.95 -> 0.80 during AM
    ("15:00", "20:00", 0.20),  # 0.80 -> 0.60 during PM
    ("20:00", "22:30", 0.05),  # 0.60 -> 0.90 evening rebound
]

PERIPH_STAY_SCHEDULE = [
    ("06:00", "10:00", 0.5),  # 0.05 -> 0.20 during AM
    ("15:00", "20:00", 0.7),  # 0.20 -> 0.40 during PM
    ("20:00", "22:30", 0.95),  # 0.40 -> 0.10 evening tighten
]


ENABLE_MASS_BASE_SCHEDULE = True
MASS_SCHEDULE_MODE = "multiplier"   # or "absolute"
MASS_BASELINE_NORMALIZE = None      # or "mean1"

CENTER_MASS_SCHEDULE = [
    ("06:00","09:00", 1.25),
    ("15:00","20:00", 0.80),
]
PERIPH_MASS_SCHEDULE = [
    ("06:00","09:00", 0.90),
    ("15:00","20:00", 1.20),
]

# =========================
# λ modulation (additive mode)
# =========================
ENABLE_ADDITIVE_LAMBDA = True

# Mass attraction: strong AM pull in, mild midday pull in, strong PM push out (via s(t) signs)
LAMBDA_DEST = 0

# Stay skew: with s=+1 (AM) periphery gets *lower* pst, center gets *higher* pst;
# with s=-1 (PM) this flips (center lowers pst → outflow).
LAMBDA_STAY = 0            # AM periphery pst ≈ 0.98 - 0.45 = 0.53 (big movement)
                              # AM center pst ≈ 0.98 + 0.45 → clamped to 0.98 (very sticky)
                              # MIDDAY periphery pst ≈ 0.98 - 0.18 = 0.80 (moderate)
                              # PM center pst ≈ 0.98 - 0.45 = 0.53 (strong outflow)

# =========================
# Diagnostics rings (optional)
# =========================
DIAGNOSTIC_RING_BINS = 8




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
def _edge_km_from_precision(p: int) -> float:
    # Mid-latitude averages; good enough for our purposes
    # (width/height are similar by p>=5; we use a single edge estimate)
    table = {
        4: 19.5,
        5: 4.9,
        6: 1.2,
        7: 0.153,
        8: 0.038,
    }
    return float(table.get(int(p), 4.9))  # fallback to 5-char default

# =========================
# Distance sampling (derived from geohash precision)
# =========================
CELL_EDGE_KM_EST   = _edge_km_from_precision(GEOHASH_PRECISION)
CELL_DIAG_KM_EST   = CELL_EDGE_KM_EST * math.sqrt(2.0)
SELF_MAX_PERIM_KM  = 4.0 * CELL_EDGE_KM_EST
R1_MAX_KM          = CELL_DIAG_KM_EST
R2_MIN_KM          = 0.75 * CELL_EDGE_KM_EST
R2_MAX_KM          = 2.0 * CELL_DIAG_KM_EST
AUTO_CENTER_PAD_KM  = 0.5 * CELL_EDGE_KM_EST  # cushion added beyond furthest center

LN_MEDIANS = {
    0: 0.25 * CELL_EDGE_KM_EST,
    1: 0.9  * CELL_EDGE_KM_EST,
    2: 1.8  * CELL_EDGE_KM_EST,
}
LN_SIGMA = 0.55

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
    if rp <= rc: rp = rc + 1e-6
    u = (r_km - rc) / (rp - rc)
    if u < 0.0: u = 0.0
    elif u > 1.0: u = 1.0
    return 1.0 - u  # g∈[0,1], ≈1 at center, ≈0 at far periphery

# =========================
# ADDITIVE mass modulation
# =========================
# =========================
# ADDITIVE mass modulation (no hard clamp; only non-negativity)
# =========================

def assign_center_periph_values_for_nodes(nodes: List[str], centers: List[str],
                                          center_val: float, periph_val: float) -> np.ndarray:
    """
    Return a vector aligned to `nodes`, assigning `center_val` to nodes in `centers`
    and `periph_val` otherwise.
    """
    centers_set = set(centers)
    vals = np.full(len(nodes), float(periph_val), dtype=float)
    for idx, g in enumerate(nodes):
        if g in centers_set:
            vals[idx] = float(center_val)
    return vals



def apply_mass_modulation(nodes, rkm_map, M_base: np.ndarray, minute_of_day: int) -> np.ndarray:
    """
    If ENABLE_ADDITIVE_LAMBDA:
        M_mod = M_base + s(t) * LAMBDA_DEST * g(r)
    else (legacy multiplicative):
        M_mod = M_base * (1 + s(t) * LAMBDA_DEST * g(r))

    Hard limits REMOVED: no epsilon floors. Only enforce non-negativity (masses cannot be <0).
    """
    s = phase_s_of_minute(minute_of_day)
    if abs(s) < 1e-12 or LAMBDA_DEST == 0.0:
        return np.maximum(M_base, 0.0)  # legality: masses ≥ 0

    gvals = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)

    if ENABLE_ADDITIVE_LAMBDA:
        M_mod = M_base + (LAMBDA_DEST * s) * gvals
    else:
        M_mod = M_base * (1.0 + (LAMBDA_DEST * s) * gvals)

    # Legality only: masses cannot be negative
    return np.maximum(M_mod, 0.0)


# =========================
# ADDITIVE stay modulation (inside build_P_for_minute)
# =========================
# =========================
# ADDITIVE stay modulation (no P_STAY_MIN/MAX; clamp only to [0,1])
# =========================
def stay_probability_for_origin(g_j: float, minute_of_day: int) -> float:
    """
    Time-varying baseline pst with optional center/periphery interpolation:
      base_center[b], base_periph[b] built once per day.
      base_pst = periph + g_j * (center - periph)   # smooth by centralness
    Then λ-modulation (additive/multiplicative) with s(t) and (2g-1).
    Final clamp to [0,1].
    """
    s = phase_s_of_minute(minute_of_day)
    center_vs_periph = (2.0 * g_j - 1.0)

    # --- baseline (scheduled) ---
    if ENABLE_CENTER_PERIPH_STAY:
        # from precomputed arrays
        bins_per_day = int(((WINDOW_END_HH - WINDOW_START_HH) * 60) // TIME_RES_MIN)
        b = int((_minute_to_bin(minute_of_day)) % max(1, bins_per_day))
        base_center = float(_CENTER_BASELINE_BY_BIN[b]) if _CENTER_BASELINE_BY_BIN is not None else float(P_STAY_BASE_MODE[0])
        base_periph = float(_PERIPH_BASELINE_BY_BIN[b]) if _PERIPH_BASELINE_BY_BIN is not None else float(P_STAY_BASE_MODE[1])
        base_pst = base_periph + g_j * (base_center - base_periph)
    else:
        # legacy single baseline
        base_pst = float(P_STAY_BASE)

    # --- modulation ---
    if ENABLE_ADDITIVE_LAMBDA:
        pst = base_pst + (LAMBDA_STAY * s) * center_vs_periph
    else:
        pst = base_pst * (1.0 + (LAMBDA_STAY * s) * center_vs_periph)

    return float(min(1.0, max(0.0, pst)))



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

def _reference_center_latlon(
    dfw: pd.DataFrame,
    nodes: List[str],
    k_for_seed: int = 25,
) -> Tuple[float, float]:
    """
    Returns (lat, lon) for the reference center used to compute r_km.
    Obeys CENTER_REFERENCE.
    """
    mode = str(CENTER_REFERENCE).lower()
    if mode == "custom" and CENTER_REF_LATLON and all(v is not None for v in CENTER_REF_LATLON):
        lat, lon = CENTER_REF_LATLON
        return float(lat), float(lon)

    if mode == "all_nodes_centroid":
        return centroid_of_tiles(nodes)

    # default: inflow centroid (if template exists) — robust anchor in practice
    try:
        seed = infer_centers_by_inflow(dfw, max(1, int(k_for_seed)))
        if seed:
            return centroid_of_tiles(seed)
    except Exception:
        pass
    # fallback: centroid of all nodes
    return centroid_of_tiles(nodes)

def select_centers(
    nodes: List[str],
    rkm_map: Dict[str, float],
    dfw: pd.DataFrame,
    mode: str,
    k: int,
    radius_km: float
) -> Tuple[List[str], Dict[str, float], dict]:
    """
    Returns:
      centers: List[str]      → chosen center tiles
      metric:  Dict[str,float]→ metric value per tile used to rank/select
      meta:    dict           → description of how selection was made (for manifest/logs)
    """
    mode = str(mode).lower()

    if mode == "rank_by_radius":
        metric = {g: float(rkm_map[g]) for g in nodes}
        centers = sorted(nodes, key=lambda g: metric[g])[:max(1, int(k))]
        return centers, metric, {"mode": mode, "metric": "radius_km", "k": int(k)}

    if mode == "radius_threshold":
        metric = {g: float(rkm_map[g]) for g in nodes}
        centers = [g for g in nodes if metric[g] <= float(radius_km)]
        if not centers:
            # guardrail: keep at least the closest tile if threshold too tight
            closest = min(nodes, key=lambda g: metric[g])
            centers = [closest]
        return centers, metric, {"mode": mode, "metric": "radius_km", "radius_km": float(radius_km)}

    if mode == "inflow_topk":
        if COUNT_COL in dfw.columns:
            s = dfw.groupby(END_COL)[COUNT_COL].sum()
        else:
            s = dfw[END_COL].value_counts()
        metric = {g: float(s.get(g, 0.0)) for g in nodes}
        centers = sorted(nodes, key=lambda g: metric[g], reverse=True)[:max(1, int(k))]
        return centers, metric, {"mode": mode, "metric": "template_inflow", "k": int(k)}

    raise ValueError(f"Unknown CENTER_SELECTION_MODE={mode}")


# =========================
# Day pattern: apply λ using g(r_km) and s(t)
# =========================
# =========================
# Legacy multiplicative mass bias (kept; no hard clamp)
# =========================
def apply_bias_to_masses(nodes: List[str], rkm_map: Dict[str,float], M_base: np.ndarray, minute_of_day: int,
                         lambda_dest: float) -> np.ndarray:
    """
    Legacy multiplicative path used elsewhere:
        factor = 1 + lambda_dest * s(t) * g(r)
        M_mod  = M_base * factor

    Hard limits REMOVED: no factor clip.
    Only enforce non-negativity of final masses.
    """
    s = phase_s_of_minute(minute_of_day)
    if abs(s) < 1e-12 or lambda_dest == 0.0:
        return np.maximum(M_base, 0.0)

    gvals  = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)
    factor = 1.0 + lambda_dest * s * gvals
    M_mod  = M_base * factor
    return np.maximum(M_mod, 0.0)

def _parse_hhmm(hhmm: str) -> int:
    """Return minutes since midnight for 'HH:MM'."""
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def _minute_to_bin(minute_of_day: int) -> int:
    """Map minute_of_day to bin index in [0, bins_per_day)."""
    rel = minute_of_day - WINDOW_START_HH * 60
    return int(rel // TIME_RES_MIN)

def _build_baseline_array_for_class(
    initial_value: float,
    segments: List[tuple[str, str, float]],
    bins_per_day: int
) -> np.ndarray:
    """
    Piecewise-linear schedule across the day, repeating daily.
    - segments: list of (start_time, end_time, target_value), times as 'HH:MM'.
    - The starting level of the first segment is the current level at its start bin.
    - Gaps: hold the last value until next segment starts.
    - Overlaps: later segments take precedence (applied in given order).
    """
    # Start flat at the initial value
    arr = np.full(bins_per_day, float(initial_value), dtype=float)

    # Convert to a mutable "current level by bin"
    # We'll apply each segment in order; overlapping segments override bins they cover.
    for (t0s, t1s, target) in segments or []:
        t0 = _parse_hhmm(t0s)
        t1 = _parse_hhmm(t1s)
        # Normalize to [0, 24h) and support wrap
        if t0 == t1:
            # zero-length => treat as instant jump at t0
            idx = _minute_to_bin(t0)
            if 0 <= idx < bins_per_day:
                arr[idx:] = float(target)  # jump for remainder of day
            continue

        # Helper to “paint” a single, non-wrapping interval
        def _apply_one(lo_min: int, hi_min: int):
            lo_bin = max(0, min(bins_per_day - 1, _minute_to_bin(lo_min)))
            hi_bin = max(0, min(bins_per_day - 1, _minute_to_bin(hi_min - 1)))  # inclusive end
            if hi_bin < lo_bin:
                return
            # Starting value at lo_bin (current arr)
            start_val = float(arr[lo_bin])
            end_val = float(target)
            span = hi_bin - lo_bin
            if span <= 0:
                # single bin -> set to target
                arr[lo_bin] = end_val
                return
            # Linear ramp from start_val -> end_val across [lo_bin..hi_bin]
            ramp = np.linspace(start_val, end_val, span + 1)
            arr[lo_bin:hi_bin + 1] = ramp

            # Hold the end_val forward until another segment overrides it
            if hi_bin + 1 < bins_per_day:
                arr[hi_bin + 1 :] = end_val

        if t0 < t1:
            _apply_one(t0, t1)
        else:
            # wrap: [t0..24:00) and [00:00..t1)
            _apply_one(t0, 24 * 60)
            _apply_one(0, t1)

    # Clamp to [0,1] for sanity (base only; final pst still clamped later)
    return np.clip(arr, 0.0, 1.0)


# =========================
# Markov builder
# =========================
# =========================
# Markov builder (no epsilon floors on Mw; keep distance epsilon only)
# =========================
def build_P_for_minute(
    nodes: List[str],
    rkm_map: Dict[str, float],
    centers_set: set,
    minute_of_day: int,
    M_base: np.ndarray,
    nb_idx: np.ndarray,
    nb_dist: np.ndarray
) -> np.ndarray:
    N, K = nb_idx.shape

    # (1) masses with additive λ
    M_mod = apply_mass_modulation(nodes, rkm_map, M_base, minute_of_day)
    Mw = np.power(np.maximum(M_mod, 0.0), ALPHA)  # legality: base non-negativity only

    # precompute g(r) & s(t) (unchanged)
    gvals = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)

    P = np.zeros((N, N), dtype=float)
    for j in range(N):
        # (2) stay prob with additive λ
        pst = stay_probability_for_origin(gvals[j], minute_of_day)
        # (3) gravity weights to neighbors
        neigh = nb_idx[j]
        distk = nb_dist[j]

        if DIST_DECAY_MODE == "power":
            # LEGALITY ONLY: avoid division by zero on distance; no floors on Mw
            denom = np.power(np.maximum(distk, 1e-12), BETA)
            w = Mw[neigh] / denom
        else:
            w = Mw[neigh] * np.exp(-GAMMA_DECAY * distk)

        # inner neighbor boost (excluding self at rank 0)
        if NEIGHBOR_INNER_K > 0 and NEIGHBOR_BOOST not in (None, 1.0, 0.0):
            inner_mask = np.zeros_like(neigh, dtype=bool)
            inner_mask[1:min(NEIGHBOR_INNER_K, len(inner_mask))] = True
            w = w * np.where(inner_mask & (neigh != j), float(NEIGHBOR_BOOST), 1.0)

        # normalize non-self and place spread
        mask = (neigh != j)
        neigh_noself = neigh[mask]
        w_noself = w[mask]
        ssum = float(np.sum(w_noself))
        if ssum <= 0.0:
            # legality: a column must be stochastic
            P[j, j] = 1.0
            continue
        w_noself = w_noself / ssum

        P[j, j] = pst
        spread = 1.0 - pst
        P[neigh_noself, j] += spread * w_noself

        # final column normalization (legality)
        colsum = P[:, j].sum()
        if colsum > 0:
            P[:, j] /= colsum
        else:
            P[j, j] = 1.0  # fallback

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


# =========================
# Initial masses (remove epsilon floor; legality only: non-negative)
# =========================
def build_initial_masses(dfw: pd.DataFrame, nodes: List[str], rkm_map,node_radii_km, centers) -> np.ndarray:
    mode = str(MASS_INIT_MODE).lower()
    if mode == "flat":
        M = np.ones(len(nodes), dtype=float)
    elif mode == "template_inflow_day0":
        M = _template_inflow_vector(dfw, nodes, mode="day0")
    else:  # "template_window_mean"
        hs = set(MASS_INIT_HOUR_SET) if MASS_INIT_HOUR_SET else None
        M = _template_inflow_vector(dfw, nodes, mode="window", hour_set=hs)

    # normalize mean for scale (unchanged)
    M = M / (M.mean() + 1e-12)

    # Optional radial reshaping (multiplicative), then renormalize mean if requested
    if MASS_INIT_MODE == "center_periph":
        M = assign_center_periph_values_for_nodes(
            nodes,
            centers,
            MASS_CENTER_PERIPH[0],
            MASS_CENTER_PERIPH[1],
        )
        # optional normalization to keep overall mass scale comparable
        M = M / (M.mean() + 1e-12)
    elif RADIAL_INIT_ENABLE_MASS:
        mult = radial_profile_multiplier_vector(
            nodes=nodes,
            rkm_map=rkm_map,
            rc_km=RADIAL_INIT_INNER_KM,
            rp_km=RADIAL_INIT_OUTER_KM,
            center_mult=RADIAL_INIT_CENTER_MULT,
            periph_mult=RADIAL_INIT_PERIPH_MULT
        )
        M = M * np.maximum(mult, 0.0)
        if RADIAL_INIT_NORMALIZE == "mean1":
            M = M / (M.mean() + 1e-12)

    # Legality only: no negative masses
    return np.maximum(M, 0.0)


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

def compute_initial_state(P_daily: List[np.ndarray], nodes: List[str], dfw: pd.DataFrame,node_radii_km, centers, rkm_map=None) -> np.ndarray:
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
    
    if mode == "center_periph":
        x0 = assign_center_periph_values_for_nodes(
            nodes,
            centers,
            X_CENTER_PERIPH[0],
            X_CENTER_PERIPH[1],
        )
        x0 = x0 / (x0.sum() + 1e-12)
        return x0


    Pm = np.mean(np.stack(P_daily, axis=0), axis=0)
    x0 = power_stationary(Pm)

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

# def write_pep_temporal_diagnostics(od: pd.DataFrame, centers: set, out_path: str) -> None:
#     """
#     Bin-level diagnostics with BOTH boundary and stay/move splits.
#     New columns:
#       - stay_total, move_total
#       - stay_center, move_center_center
#       - stay_periphery, move_periphery_periphery
#     Existing columns preserved for continuity.
#     """
#     tmp = od.copy()

#     # Ensure timestamp column
#     if "_t0" not in tmp.columns:
#         if TIME_COL not in tmp.columns:
#             raise ValueError("OD frame missing time column.")
#         tmp["_t0"] = pd.to_datetime(tmp[TIME_COL], errors="coerce", utc=False)

#     # Bin keys
#     tmp["_day"]     = tmp["_t0"].dt.floor("D")
#     tmp["_bin_idx"] = ((tmp["_t0"].dt.hour * 60 + tmp["_t0"].dt.minute) - WINDOW_START_HH * 60) // TIME_RES_MIN
#     tmp["_bin_idx"] = tmp["_bin_idx"].astype(int)
#     tmp["_bin_lab"] = tmp["_t0"].dt.strftime("%H:%M")

#     START, END = START_COL, END_COL
#     is_center_orig = tmp[START].isin(centers)
#     is_center_dest = tmp[END].isin(centers)
#     is_self = (tmp[START] == tmp[END])

#     keys = ["_day", "_bin_idx", "_bin_lab"]

#     # Boundary flows (unchanged)
#     inflow_center  = tmp.loc[~is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
#     outflow_center = tmp.loc[ is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()

#     # Old "self_*" buckets actually include stays + intra-block moves
#     self_center_old    = tmp.loc[ is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
#     self_periph_old    = tmp.loc[~is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()

#     # True stays vs moves
#     stay_total = tmp.loc[ is_self].groupby(keys)[COUNT_COL].sum()
#     move_total = tmp.loc[~is_self].groupby(keys)[COUNT_COL].sum()

#     # Split by center/periphery AND stay/move
#     stay_center = tmp.loc[ is_self &  is_center_orig].groupby(keys)[COUNT_COL].sum()
#     stay_periph = tmp.loc[ is_self & ~is_center_orig].groupby(keys)[COUNT_COL].sum()

#     move_center_center = tmp.loc[~is_self &  is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
#     move_periph_periph = tmp.loc[~is_self & ~is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()

#     total_bin = tmp.groupby(keys)[COUNT_COL].sum()

#     out = (
#         pd.DataFrame({
#             # original columns (kept)
#             "inflow_center": inflow_center,
#             "outflow_center": outflow_center,
#             "self_center": self_center_old,
#             "self_periphery": self_periph_old,
#             "total_bin_flow": total_bin,
#             # new clarity metrics
#             "stay_total": stay_total,
#             "move_total": move_total,
#             "stay_center": stay_center,
#             "stay_periphery": stay_periph,
#             "move_center_center": move_center_center,
#             "move_periphery_periphery": move_periph_periph,
#         })
#         .fillna(0)
#         .reset_index()
#     )

#     out["net_out_center"] = out["outflow_center"] - out["inflow_center"]

#     # quick consistency checks
#     # total_bin_flow == stay_total + move_total
#     # and original "self_*" == stay_* + move_* within each block
#     # (do not hard fail; just warn if off)
#     ok1 = np.allclose(out["total_bin_flow"], out["stay_total"] + out["move_total"])
#     if not ok1:
#         warn("total_bin_flow differs from stay_total+move_total for some bins.")
#     ok2_c = np.allclose(out["self_center"], out["stay_center"] + out["move_center_center"])
#     ok2_p = np.allclose(out["self_periphery"], out["stay_periphery"] + out["move_periphery_periphery"])
#     if not (ok2_c and ok2_p):
#         warn("Block totals differ from (stay+move) decomposition for some bins.")

#     out.sort_values(["_day", "_bin_idx"], inplace=True)
#     out.to_csv(out_path, index=False)


def _ensure_t0(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a _t0 timestamp column exists (local naive)."""
    if "_t0" in df.columns:
        return df
    if TIME_COL not in df.columns:
        raise ValueError("OD frame missing time column.")
    out = df.copy()
    out["_t0"] = pd.to_datetime(out[TIME_COL], errors="coerce", utc=False)
    return out

def compute_temporal_diagnostics_df(od: pd.DataFrame, centers: set) -> pd.DataFrame:
    """
    Build the per-bin 4-way partition entirely in memory and return the DataFrame.
    Columns:
      _day, _bin_idx, _bin_lab, inflow_center, outflow_center,
      self_center, self_periphery, total_bin_flow, net_out_center
    """
    tmp = _ensure_t0(od).copy()

    # Bin keys
    tmp["_day"]     = tmp["_t0"].dt.floor("D")
    tmp["_bin_idx"] = ((tmp["_t0"].dt.hour * 60 + tmp["_t0"].dt.minute)
                       - WINDOW_START_HH * 60) // TIME_RES_MIN
    tmp["_bin_idx"] = tmp["_bin_idx"].astype(int)
    tmp["_bin_lab"] = tmp["_t0"].dt.strftime("%H:%M")

    # Masks
    START, END = START_COL, END_COL
    is_center_orig = tmp[START].isin(centers)
    is_center_dest = tmp[END].isin(centers)

    # Group keys
    keys = ["_day", "_bin_idx", "_bin_lab"]

    inflow_center  = tmp.loc[~is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
    outflow_center = tmp.loc[ is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()
    self_center    = tmp.loc[ is_center_orig &  is_center_dest].groupby(keys)[COUNT_COL].sum()
    self_periphery = tmp.loc[~is_center_orig & ~is_center_dest].groupby(keys)[COUNT_COL].sum()
    total_bin      = tmp.groupby(keys)[COUNT_COL].sum()

    out = (
        pd.DataFrame({
            "inflow_center": inflow_center,
            "outflow_center": outflow_center,
            "self_center": self_center,
            "self_periphery": self_periphery,
            "total_bin_flow": total_bin,
        })
        .fillna(0)
        .reset_index()
        .sort_values(["_day", "_bin_idx"])
    )
    out["net_out_center"] = out["outflow_center"] - out["inflow_center"]

    # Optional consistency check
    part_sum = (out["inflow_center"] + out["outflow_center"]
                + out["self_center"] + out["self_periphery"])
    if not np.allclose(part_sum.values, out["total_bin_flow"].values):
        warn("Partition totals do not equal total_bin_flow for some bins (check centers set and COUNT_COL).")

    return out

def write_pep_temporal_diagnostics(od: pd.DataFrame, centers: set, out_path: str) -> None:
    df = compute_temporal_diagnostics_df(od, centers)
    df.to_csv(out_path, index=False)

def write_pep_mean_bin_balance(od: pd.DataFrame, centers: set, out_path: str) -> None:
    df = compute_temporal_diagnostics_df(od, centers)
    cols = [
        "inflow_center",
        "outflow_center",
        "self_center",
        "self_periphery",
        "total_bin_flow",
        "net_out_center",
    ]
    grp = (
        df.groupby(["_bin_idx", "_bin_lab"], as_index=False)[cols]
          .mean()
          .sort_values("_bin_idx")
    )
    grp.to_csv(out_path, index=False)

def write_pep_daily_balance(od: pd.DataFrame, centers: set, out_path: str) -> None:
    df = compute_temporal_diagnostics_df(od, centers)
    cols = [
        "inflow_center",
        "outflow_center",
        "self_center",
        "self_periphery",
        "total_bin_flow",
        "net_out_center",
    ]
    daily = (
        df.groupby("_day", as_index=False)[cols]
          .sum()
          .rename(columns={"total_bin_flow": "total_day_flow"})
          .sort_values("_day")
    )
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

def _counts_from_state(agent_state: np.ndarray, N: int) -> np.ndarray:
    return np.bincount(agent_state, minlength=N).astype(int)

def _safe_share(counts_or_sum: np.ndarray | float, total: int) -> np.ndarray | float:
    denom = float(max(1, total))
    return counts_or_sum / denom

def _reconcile_center_delta_vs_flows(tag: str):
    daily_path = os.path.join(OUTPUT_DIR, f"pep_daily_balance_{tag}.csv")
    agg_path   = os.path.join(OUTPUT_DIR, f"pep_population_aggregates_{tag}.csv")
    if not (os.path.exists(daily_path) and os.path.exists(agg_path)):
        warn("[RECON] missing files for reconciliation")
        return
    d = pd.read_csv(daily_path)
    a = pd.read_csv(agg_path)
    net = float(d["net_out_center"].sum())                 # out − in
    cen = a.loc[a["group"]=="center_set"].iloc[0]
    delta = float(cen["final_count_total"] - cen["initial_count_total"])
    if abs(delta + net) > 1e-6:
        warn(f"[RECON MISMATCH] center delta={delta:.0f}, -sum(net_out_center)={-net:.0f}, "
             f"diff={delta + net:+.0f}")
    else:
        info("[RECON] OK: center delta equals -sum(net_out_center)")

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



    # if manifest didn't store centers or it's empty, fall back to centroid of all nodes
    # center_latlon = centroid_of_tiles(list(centers) if centers else list(nodes)[:CENTER_K])
    # rkm_map = radial_km(nodes, center_latlon)

    # --- Rebuild or reuse radial distances consistent with generation ---
    # --- Rebuild or reuse radial distances consistent with generation ---
    rp_eff = float(meta.get("r_periph_sat_km", R_PERIPH_SAT_KM))
    rc_eff = float(meta.get("r_center_sat_km", R_CENTER_SAT_KM))

    # Use the actual stored centers if present; else fall back to the same logic used at generation time.
    if centers:
        center_latlon = centroid_of_tiles(list(centers))
    else:
        center_meta = meta.get("center_selection_meta", {}) or {}
        ref_latlon = meta.get("center_ref_latlon")
        mode_used = str(center_meta.get("mode", CENTER_SELECTION_MODE)).lower()
        if mode_used in ("rank_by_radius", "radius_threshold") and ref_latlon and all(isinstance(v, (int, float)) for v in ref_latlon):
            center_latlon = tuple(ref_latlon)
        else:
            center_latlon = centroid_of_tiles(list(nodes))

    rkm_map = radial_km(nodes, center_latlon)
    periphery_set = {g for g, r in rkm_map.items() if r >= rp_eff}


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

    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y%m%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y%m%d")
    tag = f"{start_tag}_{end_tag}_m{TIME_RES_MIN}"
    # Diagnostics: temporal/day-bin/daily balances
    if ENABLE_DIAG_TEMPORAL:
        td_path = os.path.join(OUTPUT_DIR, f"pep_temporal_diagnostics_{tag}.csv")
        write_pep_temporal_diagnostics(od, centers, td_path)
        info(f"[WRITE] temporal diagnostics: {td_path}")
    else:
        info("[SKIP] temporal diagnostics (disabled)")

    if ENABLE_DIAG_MEAN_BIN:
        mb_path = os.path.join(OUTPUT_DIR, f"pep_mean_bin_balance_{tag}.csv")
        write_pep_mean_bin_balance(od, centers,  mb_path)
        info(f"[WRITE] mean bin balance: {mb_path}")
    else:
        info("[SKIP] mean bin balance (disabled)")

    if ENABLE_DIAG_DAILY:
        dl_path = os.path.join(OUTPUT_DIR, f"pep_daily_balance_{tag}.csv")
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

    _reconcile_center_delta_vs_flows(tag)
    info("Validation complete.")

# =========================
# Core: generate (EVERY agent transitions & emits)
# =========================
def generate():
    banner("PEP GENERATION")
    rng = np.random.default_rng(SEED)
    bins_per_day = int(((WINDOW_END_HH - WINDOW_START_HH) * 60) // TIME_RES_MIN)
    minutes_in_day = [WINDOW_START_HH*60 + b*TIME_RES_MIN for b in range(bins_per_day)]
    # Time bins for one day (pattern)
    # --- NEW: precompute time-varying stay baselines per bin (center/periphery) ---
    global _CENTER_BASELINE_BY_BIN, _PERIPH_BASELINE_BY_BIN
    if ENABLE_STAY_BASE_SCHEDULE and ENABLE_CENTER_PERIPH_STAY:
        c0, p0 = P_STAY_BASE_MODE
        _CENTER_BASELINE_BY_BIN = _build_baseline_array_for_class(c0, CENTER_STAY_SCHEDULE, bins_per_day)
        _PERIPH_BASELINE_BY_BIN = _build_baseline_array_for_class(p0, PERIPH_STAY_SCHEDULE, bins_per_day)
    else:
        # flat baselines (either feature disabled or global non-regional mode)
        c0, p0 = P_STAY_BASE_MODE
        _CENTER_BASELINE_BY_BIN = np.full(bins_per_day, float(c0), dtype=float)
        _PERIPH_BASELINE_BY_BIN = np.full(bins_per_day, float(p0), dtype=float)


    # --- NEW: precompute time-varying MASS baselines per bin (center/periphery) ---
    global _CENTER_MASS_BASELINE_BY_BIN, _PERIPH_MASS_BASELINE_BY_BIN

    if ENABLE_MASS_BASE_SCHEDULE:
        if MASS_SCHEDULE_MODE == "multiplier":
            _CENTER_MASS_BASELINE_BY_BIN = _build_baseline_array_for_class(1.0, CENTER_MASS_SCHEDULE, bins_per_day)
            _PERIPH_MASS_BASELINE_BY_BIN = _build_baseline_array_for_class(1.0, PERIPH_MASS_SCHEDULE, bins_per_day)
        else:  # "absolute"
            c0_abs, p0_abs = MASS_CENTER_PERIPH
            _CENTER_MASS_BASELINE_BY_BIN = _build_baseline_array_for_class(c0_abs, CENTER_MASS_SCHEDULE, bins_per_day)
            _PERIPH_MASS_BASELINE_BY_BIN = _build_baseline_array_for_class(p0_abs, PERIPH_MASS_SCHEDULE, bins_per_day)
    else:
        if MASS_SCHEDULE_MODE == "multiplier":
            _CENTER_MASS_BASELINE_BY_BIN = np.ones(bins_per_day, dtype=float)
            _PERIPH_MASS_BASELINE_BY_BIN = np.ones(bins_per_day, dtype=float)
        else:  # "absolute"
            c0_abs, p0_abs = MASS_CENTER_PERIPH
            _CENTER_MASS_BASELINE_BY_BIN = np.full(bins_per_day, float(c0_abs), dtype=float)
            _PERIPH_MASS_BASELINE_BY_BIN = np.full(bins_per_day, float(p0_abs), dtype=float)

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

        ref_latlon = _reference_center_latlon(dfw, nodes, k_for_seed=max(1, int(CENTER_K)))
        rkm_map = radial_km(nodes, ref_latlon)
        # centers = infer_centers_by_inflow(dfw, CENTER_K)
        # --- Choose the "center set" per the configured metric ---
        centers, center_metric, center_meta = select_centers(
            nodes=nodes,
            rkm_map=rkm_map,
            dfw=dfw,
            mode=CENTER_SELECTION_MODE,
            k=int(CENTER_K),
            radius_km=float(CENTER_RADIUS_KM),
        )

        info(f"Center selection: mode={CENTER_SELECTION_MODE} meta={center_meta}")
        info(f"Center tiles (n={len(centers)}): {centers[:min(10, len(centers))]}{'...' if len(centers)>10 else ''}")

        # Optional: diagnostics of spread of chosen centers
        if centers:
            rvals = np.array([rkm_map[g] for g in centers], dtype=float)
            info(f"[CENTER] r_km: min={rvals.min():.2f}  median={np.median(rvals):.2f}  max={rvals.max():.2f}")

        info(f"Center tiles (K={CENTER_K}): {centers}")
    else:
        raise FileNotFoundError("Template required (USE_TEMPLATE=True).")

    # Radial distances (km) and optional diagnostic rings
    # center_latlon = centroid_of_tiles(centers)
    # rkm_map = radial_km(nodes, center_latlon)
    # _ = diagnostic_ring_labels(nodes, rkm_map, DIAGNOSTIC_RING_BINS)

    _ = diagnostic_ring_labels(nodes, rkm_map, DIAGNOSTIC_RING_BINS)

    node_radii_km = np.array([rkm_map[n] for n in nodes])

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

    # After: nb_idx, nb_dist = neighbor_indices(...)
    centers_set = set(centers)
    is_center = np.array([g in centers_set for g in nodes], dtype=bool)

    # For each origin, does it have ANY center neighbor (excluding self)?
    has_center_nb = []
    for j in range(len(nodes)):
        neigh = nb_idx[j]
        neigh = neigh[neigh != j]
        has_center_nb.append(bool(np.any(is_center[neigh])))

    frac_origins_with_center_neighbor = float(np.mean(has_center_nb))
    num_origins_with_center_neighbor  = int(np.sum(has_center_nb))
    info(f"[NEIGH] origins with >=1 CENTER neighbor: {num_origins_with_center_neighbor}/{len(nodes)} "
        f"({100*frac_origins_with_center_neighbor:.1f}%)")



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
    # Base destination masses (static pattern)
    M_base0 = build_initial_masses(dfw, nodes, rkm_map, node_radii_km, centers)
    centers_set = set(centers)
    is_center_vec = np.array([g in centers_set for g in nodes], dtype=bool)

    # Time-only mass baselines per bin (binary: center vs non-center)
    M_base_by_bin: List[np.ndarray] = []
    for b in range(bins_per_day):
        c_b = float(_CENTER_MASS_BASELINE_BY_BIN[b])
        p_b = float(_PERIPH_MASS_BASELINE_BY_BIN[b])

        if MASS_SCHEDULE_MODE == "multiplier":
            # scale the static pattern per class
            factors = np.where(is_center_vec, c_b, p_b)
            M_b = np.maximum(M_base0 * factors, 0.0)
        else:  # "absolute"
            # replace by absolute class levels
            M_b = np.where(is_center_vec, c_b, p_b).astype(float)
            M_b = np.maximum(M_b, 0.0)

        if MASS_BASELINE_NORMALIZE == "mean1":
            M_b = M_b / (M_b.mean() + 1e-12)

        M_base_by_bin.append(M_b)

    # Build daily pattern once
    # Build daily pattern once (mass varies by bin, no radial blending)
    P_daily = [
        build_P_for_minute(
            nodes=nodes,
            rkm_map=rkm_map,
            centers_set=centers_set,
            minute_of_day=mday,
            M_base=M_base_by_bin[b],
            nb_idx=nb_idx,
            nb_dist=nb_dist,
        )
        for b, mday in enumerate(minutes_in_day)
]


    # Initial state x0
    x0 = compute_initial_state(P_daily, nodes, dfw,node_radii_km, centers,rkm_map=rkm_map)
 
    is_center_mask = np.array([g in set(centers) for g in nodes], dtype=bool)
    print("[x0] sum_center:", x0[is_center_mask].sum(), "sum_noncenter:", x0[~is_center_mask].sum())

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
            "center_selection_mode": CENTER_SELECTION_MODE,
            "center_selection_meta": center_meta,
            "center_ref_latlon": ref_latlon,
            "stay_schedule_enabled": bool(ENABLE_STAY_BASE_SCHEDULE),
            "stay_initial_baselines": {"center": float(P_STAY_BASE_MODE[0]), "periphery": float(P_STAY_BASE_MODE[1])},
            "stay_center_segments": CENTER_STAY_SCHEDULE,
            "stay_periphery_segments": PERIPH_STAY_SCHEDULE,
            "time_res_min": int(TIME_RES_MIN),  # already there, but make sure it is
            "mass_schedule_enabled": bool(ENABLE_MASS_BASE_SCHEDULE),
            "mass_schedule_mode": MASS_SCHEDULE_MODE,
            "mass_baseline_normalize": MASS_BASELINE_NORMALIZE,
            "center_mass_segments": CENTER_MASS_SCHEDULE,
            "periphery_mass_segments": PERIPH_MASS_SCHEDULE,

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


    if DETERMINISTIC_INITIAL_ASSIGNMENT:
        # Compute expected counts
        expected = x0 * POOL_SIZE

        # Floor to integers
        counts = np.floor(expected).astype(int)
        remainder = POOL_SIZE - counts.sum()

        # Distribute leftover agents to tiles with largest fractional parts
        if remainder > 0:
            fractional = expected - counts
            top_idx = np.argsort(-fractional)[:remainder]
            counts[top_idx] += 1

        # Build the agent_state deterministically
        agent_state = np.repeat(np.arange(N), counts)
        # Optional: shuffle for randomness of order, not counts
        rng.shuffle(agent_state)

        # Also store the initial per-tile population
        initial_counts = counts
    else:
        agent_state = rng.choice(N, size=POOL_SIZE, p=x0, replace=True)
        initial_counts = np.bincount(agent_state, minlength=N)


    # initial positions from x0
    # agent_state = rng.choice(N, size=POOL_SIZE, p=x0, replace=True)
    initial_agent_state = agent_state.copy()  # <-- capture initial

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

    # --- NEW: emit daily baseline schedule for debug ---
    try:
        labs = [f"{(WINDOW_START_HH*60 + b*TIME_RES_MIN)//60:02d}:{(WINDOW_START_HH*60 + b*TIME_RES_MIN)%60:02d}" for b in range(bins_per_day)]
        sched_df = pd.DataFrame({
            "_bin_idx": np.arange(bins_per_day, dtype=int),
            "_bin_lab": labs,
            "center_baseline": _CENTER_BASELINE_BY_BIN if _CENTER_BASELINE_BY_BIN is not None else [],
            "periph_baseline": _PERIPH_BASELINE_BY_BIN if _PERIPH_BASELINE_BY_BIN is not None else [],
        })
        sched_path = os.path.join(OUTPUT_DIR, f"pep_stay_baseline_schedule_{tag}.csv")
        sched_df.to_csv(sched_path, index=False)
        info(f"[WRITE] stay baseline schedule (per bin): {sched_path}")
    except Exception as _e:
        warn(f"[STAY-SCHED] failed to write schedule CSV: {_e}")

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
    def write_population_reports(
        nodes: List[str],
        centers: List[str],
        rkm_map: Dict[str, float],
        initial_counts: np.ndarray,
        final_counts: np.ndarray,
        pool_size: int,
        out_dir: str,
        tag: str,
    ) -> None:
        """
        Writes:
        A) pep_population_by_tile_{tag}.csv
            geohash5, initial_count/share, final_count/share, delta_share,
            r_km, is_center, is_periphery_r_ge_rp, is_inner_r_lt_rc

        B) pep_population_aggregates_{tag}.csv
            groups: center_set, non_center,
                    inner_r<rc, mid_rc<=r<rp, periphery_r>=rp, TOTAL
            For each group: initial_count_total, final_count_total,
                            initial_total_share, final_total_share, delta_total_share,
                            initial_mean_count_per_tile, final_mean_count_per_tile
        """
        N = len(nodes)
        if initial_counts.shape[0] != N or final_counts.shape[0] != N:
            raise ValueError("initial_counts/final_counts have wrong length")

        # Per-tile shares and deltas
        init_share = _safe_share(initial_counts, pool_size) * 100.0
        final_share = _safe_share(final_counts, pool_size) * 100.0
        delta_share = final_share - init_share


        centers_set = set(centers)
        is_center = np.array([g in centers_set for g in nodes], dtype=bool)
        r_km = np.array([float(rkm_map[g]) for g in nodes], dtype=float)
        is_inner = r_km < float(R_CENTER_SAT_KM)
        is_periph = r_km >= float(R_PERIPH_SAT_KM)

        # -------- A) Per-tile table
        df_tiles = pd.DataFrame({
            "geohash5": nodes,
            "initial_count": initial_counts,
            "initial_share": init_share,
            "final_count": final_counts,
            "final_share": final_share,
            "delta_share": delta_share,
            "r_km": r_km,
            "is_center": is_center,
            "is_periphery_r_ge_rp": is_periph,
            "is_inner_r_lt_rc": is_inner,
        }).sort_values(["is_center", "final_count"], ascending=[False, False], ignore_index=True)

        path_tiles = os.path.join(out_dir, f"pep_population_by_tile_{tag}.csv")
        df_tiles.to_csv(path_tiles, index=False)
        info(f"[WRITE] per-tile initial/final population: {path_tiles}")

        # -------- B) Aggregates
        # convenience
        def group_mask(name: str) -> np.ndarray:
            if name == "center_set":       return is_center
            if name == "non_center":       return ~is_center
            if name.startswith("inner_"):  return is_inner
            if name.startswith("mid_"):    return (~is_inner) & (~is_periph)
            if name.startswith("periphery"):return is_periph
            if name == "TOTAL":            return np.ones(N, dtype=bool)
            raise ValueError(name)

        groups = [
            "center_set",
            "non_center",
            f"inner_r<rc (rc={R_CENTER_SAT_KM}km)",
            f"mid_rc<=r<rp (rp={R_PERIPH_SAT_KM}km)",
            "periphery_r>=rp",
            "TOTAL",
        ]

        rows = []
        for gname in groups:
            m = group_mask(gname)
            # totals
            init_tot = int(initial_counts[m].sum())
            final_tot = int(final_counts[m].sum())
            init_share_tot = _safe_share(init_tot, pool_size) * 100.0
            final_share_tot = _safe_share(final_tot, pool_size) * 100.0


            # means per tile within the group
            tiles_in_g = int(m.sum())
            init_mean = float(initial_counts[m].mean()) if tiles_in_g > 0 else 0.0
            final_mean = float(final_counts[m].mean()) if tiles_in_g > 0 else 0.0

            rows.append({
                "group": gname,
                "tiles_in_group": tiles_in_g,
                "initial_count_total": init_tot,
                "final_count_total": final_tot,
                "initial_total_share": init_share_tot,
                "final_total_share": final_share_tot,
                "delta_total_share": final_share_tot - init_share_tot,
                "initial_mean_count_per_tile": init_mean,
                "final_mean_count_per_tile": final_mean,
            })

        df_agg = pd.DataFrame(rows)
        path_agg = os.path.join(out_dir, f"pep_population_aggregates_{tag}.csv")
        df_agg.to_csv(path_agg, index=False)
        info(f"[WRITE] aggregates (initial vs final): {path_agg}")

    # --- Write neighbor diagnostics (if enabled) ---
    # --- NEW: initial vs final per-tile counts + aggregates ---

    initial_counts = _counts_from_state(initial_agent_state, N)
    final_counts   = _counts_from_state(agent_state, N)

    write_population_reports(
        nodes=nodes,
        centers=centers,
        rkm_map=rkm_map,
        initial_counts=initial_counts,
        final_counts=final_counts,
        pool_size=POOL_SIZE,
        out_dir=OUTPUT_DIR,
        tag=tag,
    )


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
