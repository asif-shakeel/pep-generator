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
SWEEP_END_DATE   = "2019-07-14"   # inclusive
WINDOW_START_HH  = 0               # inclusive
WINDOW_END_HH    = 24              # exclusive
TIME_RES_MIN     = 45              # bin size (minutes)

# =========================
# Files / plots / randomness (RESTORED)
# =========================
WRITE_GZIP             = False   # legacy name; maps to COMPRESS_OUTPUTS
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
# Gravity & distance model (RESTORED semantics)
# =========================
ALPHA          = 0.70    # mass exponent (maps to ALPHA_MASS)
BETA           = 2.8     # power distance decay exponent
USE_EXP_DECAY  = False   # if True, use exp(-d / DIST_SCALE_KM)
DIST_SCALE_KM  = 5.0     # only used when USE_EXP_DECAY=True ⇒ gamma = 1/DIST_SCALE_KM
HARD_CUTOFF_KM = None    # if not None, neighbors with d > cutoff have 0 weight (except self)

# Derived internal names
ALPHA_MASS  = ALPHA
BETA_DIST   = BETA
GAMMA_DECAY = 0.0 if not USE_EXP_DECAY else (0.0 if DIST_SCALE_KM in (None, 0) else 1.0/float(DIST_SCALE_KM))
DIST_DECAY_MODE = "exp" if USE_EXP_DECAY else "power"

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
# λ-based AM/PM modulation with single g(r) in KM
# =========================
AM_HOURS_SET      = {6, 9}            # AM ramp: -1 → +1
MIDDAY_HOURS_SET  = {12}              # plateau: +1
PM_HOURS_SET      = {15, 18, 21}      # PM ramp: +1 → -1

LAMBDA_DEST = 0.35  # replaces RING_LAMBDA_DEST
LAMBDA_STAY = 0.00  # replaces RING_LAMBDA_ORIG (kept off by default to match original)

R_CENTER_SAT_KM = 8    # r_c: center plateau radius (km)
R_PERIPH_SAT_KM = 30.0   # r_p: periphery plateau radius (km)



# Optional diagnostics-only bucketing for reports (does not affect g(r))
DIAGNOSTIC_RING_BINS = 8

# =========================
# Centers & stays (RESTORED)
# =========================
CENTER_K = 6
P_STAY_BASE = 0.60
STAY_MULT_BY_HOUR     = {}  # neutral unless overridden (legacy name)
P_STAY_HOURLY_MULT    = STAY_MULT_BY_HOUR  # alias used internally
P_STAY_CENTER_BY_HOUR = {}  # optional: per-hour multiplier for center tiles only
P_STAY_PERIPH_BY_HOUR = {}  # optional: per-hour multiplier for periphery tiles only

# =========================
# Neighbor scheme & parameters (RESTORED defaults)
# =========================
NEIGHBOR_SCHEME     = "lattice"   # {"knn","lattice"}
NEIGHBOR_K          = 24          # hard cap
NEIGHBOR_INNER_K    = 8           # ring-1 threshold for inner boost & hop labels
NEIGHBOR_BOOST      = 8.0         # multiplier for inner neighbors (excluding self)
NEIGHBOR_MAX_RINGS  = 2           # lattice mode: max rings to include

# =========================
# Per-bin totals (RESTORED controllers)
# =========================
MATCH_PER_BIN_TARGET = False
TOTAL_PER_BIN_MODE   = "fixed"     # {"fixed","template_mean","series"}
TOTAL_PER_BIN_FIXED  = 12000       # **total number of agents** in the system
SERIES_SOURCE        = "first_day" # {"first_day"}
SERIES_SPEC          = (0, 60, 24)  # (offset, scale, hours)
SERIES_HOUR_SET      = None         # optional explicit hour multipliers dict {hour:int -> mult:float}

# =========================
# Initial mass & initial state configuration
# =========================
# Initial mass for destinations (hourly base BEFORE Markov normalization)
MASS_INIT_MODE     = "template_window_mean"  # {"flat","template_inflow_day0","template_window_mean"}
MASS_INIT_HOUR_SET = None  # optional: restrict window mean to these hours (set[int])

# Initial state (starting distribution over origins for agents)
INIT_X_MODE = "template_inflow_day0"  # {"flat","template_inflow_day0","template_window_mean","stationary_meanP","periodic_fixed_point"}

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


# --- Auto-tune options for saturation radii (center/periphery) ---
AUTO_TUNE_SAT_RADII = True          # enable auto tuning
AUTO_TUNE_MODE      = "floor"       # {"floor","replace"}
AUTO_PERIPH_Q       = 0.92          # percentile of all r_km for periphery plateau
AUTO_CENTER_PAD_KM  = 0.5 * CELL_EDGE_KM_EST  # cushion added beyond furthest center

# Optional hard overrides (None = unused)
RC_OVERRIDE_KM = None
RP_OVERRIDE_KM = None

# # ====== TWEAKS: stronger AM/PM tides while keeping balance ======
# LAMBDA_DEST = 1.0        # big amplitude (0.9–1.2 is a good search range)

# # Steeper center vs periphery contrast
# R_CENTER_SAT_KM  = 1.0
# R_PERIPH_SAT_KM  = 30.0

# # More willingness to move
# P_STAY_BASE = 0.40
# LAMBDA_STAY = 0.00   # keep simple; use hourly multipliers instead

# # Periphery moves more in AM, settles more in PM; centers linger a touch in AM/midday
# P_STAY_PERIPH_BY_HOUR = {6:0.75, 7:0.75, 8:0.75, 9:0.80, 15:1.20, 16:1.25, 17:1.25, 18:1.20, 21:1.15}
# P_STAY_CENTER_BY_HOUR = {8:1.10, 9:1.10, 12:1.05}

# # Make long jumps feasible and masses matter more
# NEIGHBOR_K   = 64
# BETA         = 2.0     # or: USE_EXP_DECAY=True; DIST_SCALE_KM=10.0
# ALPHA        = 0.90
# # USE_EXP_DECAY = True
# # DIST_SCALE_KM = 10.0
# # ================================================================


# ==== TWEAKS: drive long-run concentration to center (~80% target) ====
# Turn off diurnal bias while we shape the long-run drift.
LAMBDA_DEST = 0.0
LAMBDA_STAY = 0.0

# Keep people moving overall so the drift can manifest.
P_STAY_BASE = 0.25   # lower base -> more movement per bin

# MUCH leakier periphery, stickier center, but avoid hitting 0.95 frequently
# Effective examples: center pst ≈ 0.25 * 1.40 = 0.35; periphery pst ≈ 0.25 * 0.12 = 0.03
P_STAY_CENTER_BY_HOUR  = {h: 1.40 for h in range(24)}   # modest >1, not saturating
P_STAY_PERIPH_BY_HOUR  = {h: 0.12 for h in range(24)}   # much lower, strong leak

# Strengthen mass attraction & reduce distance penalty so center wins multinomials
ALPHA      = 1.30     # mass sensitivity up (big destinations win more)
BETA       = 1.50     # distance penalty down (center visible from farther out)
NEIGHBOR_K = 96       # many options -> center is in range from most tiles
HARD_CUTOFF_KM = None

# Start from data so you can see the drift
INIT_X_MODE     = "template_inflow_day0"
MASS_INIT_MODE  = "template_window_mean"
# If template has bigger center inflow in business hours, this emphasizes it:
# MASS_INIT_HOUR_SET = set(range(6, 20))  # optional

# If you also want a larger "center set" for the metric itself:
# CENTER_K = 8   # optional; makes the 'center_set' aggregate cover more tiles

# Optional (only if you want the center set to be a bit larger):
CENTER_K = 8  # (default 6) expanding the “center set” can help reach 80% faster

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
    am0, am1 = _hour_bounds_from_set(AM_HOURS_SET)
    md0, md1 = _hour_bounds_from_set(MIDDAY_HOURS_SET)
    pm0, pm1 = _hour_bounds_from_set(PM_HOURS_SET)

    if am0 is None or pm0 is None or md0 is None:
        minutes = 24*60
        phase  = (minute_of_day - 8*60 - 30) / minutes
        return float(math.sin(2*math.pi*phase))

    m = minute_of_day
    am_start = am0*60; am_end = am1*60
    md_start = md0*60; md_end = md1*60
    pm_start = pm0*60; pm_end = pm1*60

    if am_start <= m <= am_end:
        if am_end == am_start:
            return +1.0
        frac = (m - am_start) / max(1, (am_end - am_start))
        return -1.0 + 2.0*frac
    if md_start <= m <= md_end:
        return +1.0
    if pm_start <= m <= pm_end:
        if pm_end == pm_start:
            return -1.0
        frac = (m - pm_start) / max(1, (pm_end - pm_start))
        return +1.0 - 2.0*frac
    if m < am_start:
        return -1.0
    if m > pm_end:
        return -1.0
    if am_end < m < md_start:
        return +1.0
    if md_end < m < pm_start:
        return +1.0
    return -1.0

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
def _knn_neighbor_indices(nodes: List[str], K: int) -> Tuple[np.ndarray, np.ndarray]:
    N = len(nodes)
    latlons = np.array([geohash_decode(g) for g in nodes], dtype=float)
    dmat = np.zeros((N, N), dtype=float)
    for j in range(N):
        latj, lonj = latlons[j]
        for i in range(j+1, N):
            lati, loni = latlons[i]
            d = haversine_km(latj, lonj, lati, loni)
            dmat[j,i] = dmat[i,j] = d
    nb_idx = np.argpartition(dmat, kth=np.minimum(K-1, N-1), axis=0)[:K, :].T  # (N,K)
    nb_dist = np.take_along_axis(dmat, nb_idx, axis=1)
    for j in range(N):
        row = nb_idx[j]
        if j not in row:
            far = np.argmax(nb_dist[j])
            row[far] = j
            nb_dist[j, far] = 0.0
        order = np.argsort(nb_dist[j])
        nb_idx[j] = row[order]
        nb_dist[j] = nb_dist[j][order]
    return nb_idx.astype(int), nb_dist.astype(float)

def _lattice_neighbor_indices(nodes: List[str], K_cap: int, max_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    # Approximate lattice by nearest-K; ring semantics used later via ranks
    return _knn_neighbor_indices(nodes, K_cap)

def neighbor_indices(nodes: List[str], K: int, scheme: str, max_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    if scheme == "knn":
        return _knn_neighbor_indices(nodes, K)
    else:
        return _lattice_neighbor_indices(nodes, K, max_rings)

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
    Mw = np.power(M_mod, ALPHA_MASS)

    h = (minute_of_day // 60) % 24
    P = np.zeros((N, N), dtype=float)

    gvals = np.array([g_of_r(rkm_map[g], R_CENTER_SAT_KM, R_PERIPH_SAT_KM) for g in nodes], dtype=float)
    s = phase_s_of_minute(minute_of_day)

    periph_set = {g for g in nodes if rkm_map[g] >= R_PERIPH_SAT_KM}

    for j in range(N):
        g_j = gvals[j]
        pst_mult = float(P_STAY_HOURLY_MULT.get(h, 1.0))
        node_j = nodes[j]
        if node_j in centers_set:
            pst_mult *= float(P_STAY_CENTER_BY_HOUR.get(h, 1.0))
        if node_j in periph_set:
            pst_mult *= float(P_STAY_PERIPH_BY_HOUR.get(h, 1.0))

        pst = float(P_STAY_BASE * pst_mult * (1.0 - LAMBDA_STAY * s * g_j))
        pst = float(np.clip(pst, 0.0, 0.95))

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
                w = np.power(np.maximum(Mw[neigh], 1e-12), 1.0) / np.power(np.maximum(distk, 1e-6), BETA_DIST)
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
    med = LN_MEDIANS.get(int(hop_group), LN_MEDIANS[2]); sig = LN_SIGMA
    if hop_group == 0:
        lo, hi = 0.0, SELF_MAX_PERIM_KM
    elif hop_group == 1:
        lo, hi = CELL_EDGE_KM_EST, R1_MAX_KM
    else:
        lo, hi = R2_MIN_KM, R2_MAX_KM
    sampler = lambda: _sample_lognormal_km(med, sig, rng)
    return _truncate_or_resample(sampler(), lo, hi, rng, sampler)

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
# Per-bin totals controller (kept for compatibility; not used to subsample)
# =========================
def compute_target_total_for_bin(template_df: pd.DataFrame, minute_of_day: int) -> int:
    if TOTAL_PER_BIN_MODE == "fixed":
        return int(TOTAL_PER_BIN_FIXED)

    if TOTAL_PER_BIN_MODE == "template_mean":
        if COUNT_COL in template_df.columns and TIME_COL in template_df.columns:
            counts = template_df.groupby(TIME_COL)[COUNT_COL].sum()
            if not counts.empty:
                return int(round(counts.mean()))
        return int(TOTAL_PER_BIN_FIXED)

    if TOTAL_PER_BIN_MODE == "series":
        offset, scale, hours = SERIES_SPEC
        h = (minute_of_day // 60) % 24
        if SERIES_HOUR_SET and isinstance(SERIES_HOUR_SET, dict) and h in SERIES_HOUR_SET:
            base = float(SERIES_HOUR_SET[h])
        else:
            if SERIES_SOURCE == "first_day" and COUNT_COL in template_df.columns and TIME_COL in template_df.columns:
                td0 = pd.to_datetime(template_df[TIME_COL], errors="coerce", utc=False)
                d0 = td0.dt.floor('D').min()
                sel = template_df[td0.dt.floor('D') == d0]
                prof = sel.groupby(td0.dt.hour)[COUNT_COL].sum()
                base = float(prof.get(h, prof.mean() if not prof.empty else 1.0))
                if not np.isfinite(base) or base <= 0:
                    base = 1.0
                base /= float(prof.mean() if not prof.empty else 1.0)
            else:
                base = 1.0
        return int(max(0, round(offset + scale * base)))

    return int(TOTAL_PER_BIN_FIXED)

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

def build_initial_masses(dfw: pd.DataFrame, nodes: List[str]) -> np.ndarray:
    mode = str(MASS_INIT_MODE).lower()
    if mode == "flat":
        M = np.ones(len(nodes), dtype=float)
    elif mode == "template_inflow_day0":
        M = _template_inflow_vector(dfw, nodes, mode="day0")
    else:  # "template_window_mean"
        hs = set(MASS_INIT_HOUR_SET) if MASS_INIT_HOUR_SET else None
        M = _template_inflow_vector(dfw, nodes, mode="window", hour_set=hs)
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

def compute_initial_state(P_daily: List[np.ndarray], nodes: List[str], dfw: pd.DataFrame) -> np.ndarray:
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

    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(OUTPUT_DIR, "pep_model_empirical_compare.csv")
    comp_df.to_csv(comp_path, index=False)
    info(f"[WRITE] model vs empirical (per-bin, mass-weighted): {comp_path}")

    # Diagnostics: temporal/day-bin/daily balances
    td_path = os.path.join(OUTPUT_DIR, "pep_temporal_diagnostics.csv")
    mb_path = os.path.join(OUTPUT_DIR, "pep_mean_bin_balance.csv")
    dl_path = os.path.join(OUTPUT_DIR, "pep_daily_balance.csv")
    write_pep_temporal_diagnostics(od, centers, td_path)
    write_pep_mean_bin_balance(od, centers, mb_path)
    write_pep_daily_balance(od, centers, dl_path)
    info(f"[WRITE] temporal diagnostics: {td_path}")
    info(f"[WRITE] mean bin balance: {mb_path}")
    info(f"[WRITE] daily balance: {dl_path}")

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

    # Neighbors
    nb_idx, nb_dist = neighbor_indices(nodes, NEIGHBOR_K, NEIGHBOR_SCHEME, NEIGHBOR_MAX_RINGS)

    # Base destination masses
    M_base = build_initial_masses(dfw, nodes)

    # Build daily pattern once
    P_daily = [build_P_for_minute(nodes, rkm_map, set(centers), mday, M_base, nb_idx, nb_dist)
               for mday in minutes_in_day]

    # Initial state x0
    x0 = compute_initial_state(P_daily, nodes, dfw)

    # Persist transitions pattern
    trans_npy, manifest = transitions_paths()
    write_transitions(P_daily, nodes, manifest, trans_npy,
                      extra_meta={"centers": centers, "seed": SEED,
                                  "mass_init_mode": MASS_INIT_MODE,
                                  "init_x_mode": INIT_X_MODE})

    # Agent pool (ALL agents)
    N = len(nodes)
    node_to_idx = {g:i for i,g in enumerate(nodes)}
    POOL_SIZE = int(TOTAL_PER_BIN_FIXED)   # total number of agents
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

    raw_path = os.path.join(OUTPUT_DIR, f"pep_raw_{tag}.csv")
    if COMPRESS_OUTPUTS:
        raw_path += ".gz"
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            pep_df.to_csv(f, index=False)
    else:
        pep_df.to_csv(raw_path, index=False)
    info(f"Wrote raw PEP rows: {raw_path}")

    if WRITE_OD_AGGREGATE:
        g = pep_df.groupby([START_COL, END_COL, DATE_COL, TIME_COL], as_index=False)
        od = g.agg(trip_count=("length_m", "size"),
                   m_length_m=("length_m", "mean"),
                   mdn_length_m=("length_m", "median"))
        od_path = os.path.join(OUTPUT_DIR, f"pep_od_{tag}.csv")
        if COMPRESS_OUTPUTS:
            od_path += ".gz"
            with gzip.open(od_path, "wt", encoding="utf-8") as f:
                od.to_csv(f, index=False)
        else:
            od.to_csv(od_path, index=False)
        info(f"Wrote OD aggregate: {od_path}")

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
