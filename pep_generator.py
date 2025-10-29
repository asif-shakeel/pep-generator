#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pep_generator.py — Scheduled directional bias, geometric center selection, pairwise distance envelopes,
and random-bucket assignment for mass and initial population.

Highlights
- No radial g(r) modulation. "Center" is a classification only (selection modes below).
- Directional bias is scheduled with sticky (hold) semantics and end-exclusive segments.
- Pairwise distance envelopes (per origin→destination neighbor) for realistic hop-length sampling.
- Neighbor schemes: lattice or knn (with inner-neighbor boost).
- Center selection modes:
    CENTER_SELECTION_MODE ∈ {"rank_by_radius","radius_threshold","inflow_topk","fixed_ids","file"}
    CENTER_REFERENCE     ∈ {"inflow_centroid","all_nodes_centroid","custom"}
- Random-bucket assignment modes:
    MASS_ASSIGNMENT_MODE ∈ {"uniform","random_buckets"}
    X_ASSIGNMENT_MODE    ∈ {"uniform","random_buckets"}
- Manifest/run tags reflect schedules for reproducibility.
- Aggregates CSV includes initial/final totals and shares.

This file is self-contained. You can import generate()/validate() or run the file directly.
"""
from __future__ import annotations
import os
import json
import math
from typing import Dict, Tuple, List, Optional, Iterable, Union

import numpy as np
import pandas as pd

# =========================
# Mode
# =========================
MODE = "generate"  # {"generate","validate"}

# =========================
# Paths & template
# =========================
BASE_DIR   = "/Users/asif/Documents/nm24"  # or: os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RUNS_DIR   = os.path.join(OUTPUT_DIR, "runs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Template discovery
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
TIME_RES_MIN     = 180              # bin size (minutes)

# =========================
# Files / randomness
# =========================
WRITE_RAW                    = True
WRITE_OD_AGGREGATE           = True
SAVE_TRANSITIONS             = True
VALIDATE_ON_GENERATE         = True
MAKE_PLOTS                   = False
SEED                         = 1234

# =========================
# Geohash precision (fixed 5)
# =========================
GEOHASH_PRECISION = 5

# =========================
# Region filter
# =========================
COUNTRY = "Mexico"
REGION_NAME = "Mexico City, Mexico"
REGION_BBOXES = {
    "Mexico City, Mexico": (-99.45465164940843, 18.96381286042833, -98.85051749282216, 19.677365270912603),
}
REGION_ENDPOINT_RULE = "either"     # {"both","either"}
BUFFER_KM_BY_COUNTRY = {"Colombia":10.0, "Indonesia":10.0, "Mexico":10.0, "India":10.0}
USE_REGION_FILTER = True
REGION_MEMBERSHIP_MODE = "intersects"  # {"intersects","within"}

# =========================
# Gravity & distance model (unchanged core)
# =========================
ALPHA          = 0.50            # mass exponent
BETA           = 1.8             # distance decay exponent (power law)
DIST_DECAY_MODE = "power"        # {"power","exp"}
GAMMA_DECAY     = 0.0            # only if DIST_DECAY_MODE=="exp"

# =========================
# Neighbor graph
# =========================
NEIGHBOR_SCHEME     = "lattice"  # {"lattice","knn"}
NEIGHBOR_K          = 9          # total neighbor cap (including self)
NEIGHBOR_INNER_K    = 9          # inner subset boosted
NEIGHBOR_BOOST      = 4.0        # multiplicative boost for inner neighbors
NEIGHBOR_MAX_RINGS  = 3          # lattice extent (in rings)
NEIGHBOR_DISTANCE_CUTOFF_KM = None
ENABLE_NEIGHBOR_DIAGNOSTICS = False

# =========================
# Center selection (geometric + inflow + fixed/file)
# =========================
CENTER_SELECTION_MODE = "rank_by_radius"  # {"rank_by_radius","radius_threshold","inflow_topk","fixed_ids","file"}
CENTER_REFERENCE      = "all_nodes_centroid"  # {"inflow_centroid","all_nodes_centroid","custom"}
CENTER_REF_LATLON     = (19.4326, -99.1332)  # used if CENTER_REFERENCE=="custom"  (lat, lon)
CENTER_RADIUS_KM      = 3.0   # used if mode=="radius_threshold"
CENTER_TOPK           = 3     # used if mode in {"rank_by_radius","inflow_topk"}
CENTER_FIXED_IDS      = []    # used if mode=="fixed_ids"
CENTER_FILE_PATH      = os.path.join(DATA_DIR, "center_ids.txt")  # file with one geohash5 per line if mode=="file"

# =========================
# Stay probabilities (center/periphery, flat or scheduled)
# =========================
P_STAY_BASE_MODE = (0.1, 0.9)  # (center, periphery) if no schedule
ENABLE_STAY_BASE_SCHEDULE = True
STAY_SCHEDULE_MODE = "absolute"    # {"absolute","multiplier"}
CENTER_STAY_SCHEDULE = [
    ("06:00", "09:00", 0.9),
    ("15:00", "21:00", 0.1),
]
PERIPH_STAY_SCHEDULE = [
    ("06:00", "09:00", 0.1),
    ("15:00", "21:00", 0.9),
]
# Optional AM/PM tilt; +s(t) increases center, decreases periphery
LAMBDA_STAY = 0.0
# P_STAY_BASE_MODE = (0.1, 0.9)  # (center, periphery) if no schedule
# ENABLE_STAY_BASE_SCHEDULE = True
# STAY_SCHEDULE_MODE = "absolute"    # {"absolute","multiplier"}
# CENTER_STAY_SCHEDULE = [
#     ("06:00", "09:00", 0.9),
#     ("15:00", "21:00", 0.1),
# ]
# PERIPH_STAY_SCHEDULE = [
#     ("06:00", "09:00", 0.1),
#     ("15:00", "21:00", 0.9),
# ]
# # Optional AM/PM tilt; +s(t) increases center, decreases periphery
# LAMBDA_STAY = 0.0

# =========================
# Destination "mass" levels (center/periphery, flat or scheduled)
# =========================
MASS_INIT_MODE = "center_periph"  # {"flat","template_inflow_day0","template_window_mean","center_periph"}
MASS_CENTER_PERIPH = (20.0, 1.0)   # if MASS_INIT_MODE=="center_periph"

ENABLE_MASS_BASE_SCHEDULE = True
MASS_SCHEDULE_MODE = "absolute"    # {"absolute","multiplier"}
MASS_TIME_MODE = "absolute"       # schedule defines absolute levels by group
MASS_BASE_MODE = (1.0, 20.0)      # starting levels for schedule painting (center, periphery)
CENTER_MASS_SCHEDULE = [
    ("06:00", "09:00", 20.0),
    ("15:00", "21:00", 1.0),
]
PERIPH_MASS_SCHEDULE = [
    ("06:00", "09:00", 1.0),
    ("15:00", "21:00", 20.0),
]

# MASS_INIT_MODE = "center_periph"  # {"flat","template_inflow_day0","template_window_mean","center_periph"}
# MASS_CENTER_PERIPH = (20.0, 1.0)   # if MASS_INIT_MODE=="center_periph"

# ENABLE_MASS_BASE_SCHEDULE = True
# MASS_SCHEDULE_MODE = "absolute"    # {"absolute","multiplier"}
# MASS_TIME_MODE = "absolute"        # kept for backward-compat (ignored if MASS_SCHEDULE_MODE provided)
# MASS_BASE_MODE = (1.0, 20.0)       # starting levels for schedule painting (center, periphery)
# CENTER_MASS_SCHEDULE = [
#     ("06:00", "09:00", 20.0),
#     ("15:00", "21:00", 1.0),
# ]
# PERIPH_MASS_SCHEDULE = [
#     ("06:00", "09:00", 1.0),
#     ("15:00", "21:00", 20.0),
# ]

# ===== Random-bucket assignment for Mass =====
MASS_ASSIGNMENT_MODE = "uniform"  # {"uniform","random_buckets"}
# Dicts: {percent_of_tiles: percent_of_group_mass} — keys and values each sum to 100, none zero
MASS_BUCKETS_CENTER: Dict[int,int] = {10: 70, 90: 30}
MASS_BUCKETS_PERIPH: Dict[int,int] = {20: 50, 80: 50}
MASS_RANDOM_SEED: Optional[int] = None  # defaults to SEED if None

# =========================
# Initial population X (who starts where)
# =========================
INIT_X_MODE = "center_periph"  # {"flat","template_inflow_day0","template_window_mean","stationary_meanp","periodic_fixed_point","center_periph"}
X_CENTER_PERIPH = (1.0, 10.0)  # used if INIT_X_MODE=="center_periph"
DETERMINISTIC_INITIAL_ASSIGNMENT = True

# ===== Random-bucket assignment for x0 =====
X_ASSIGNMENT_MODE = "uniform"  # {"uniform","random_buckets"}
X_BUCKETS_CENTER: Dict[int,int] = {20: 60, 80: 40}
X_BUCKETS_PERIPH: Dict[int,int] = {10: 20, 90: 80}
X_RANDOM_SEED: Optional[int] = None  # defaults to SEED if None

# =========================
# Directional bias (scheduled multipliers)
# =========================
ENABLE_DIRECTIONAL_BIAS = True
DIRBIAS_MODE = "scheduled"     # {"scheduled","fixed","phased"}
DIRBIAS_SCHEDULE_MODE = "absolute"  # {"absolute","multiplier"}
DIRBIAS_BASE = (1.0, 1.0)      # used only if "multiplier"

# Schedules are piecewise-linear across bins; values are multipliers (>=0)
DIRBIAS_IN_SCHEDULE = [
    ("06:00","09:00", 10),   # periph -> center boost AM
    ("15:00","21:00", 1),   # taper evening
]
DIRBIAS_OUT_SCHEDULE = [
    ("06:00","09:00", 1),   # dampen center -> periph AM
    ("15:00","21:00", 10),   # boost evening egress
    ("21:00","00:00", 1),   # boost evening egress
]

# ENABLE_DIRECTIONAL_BIAS = True
# DIRBIAS_MODE = "scheduled"     # {"scheduled","fixed","phased"}
# DIRBIAS_SCHEDULE_MODE = "absolute"  # {"absolute","multiplier"}
# DIRBIAS_BASE = (1.0, 1.0)      # base curves for scheduled mode

# # Schedules are piecewise-linear across bins; values are multipliers (>=0)
# DIRBIAS_IN_SCHEDULE = [
#     ("06:00","09:00", 10),   # periph -> center boost AM
#     ("15:00","21:00", 1),    # taper evening
# ]
# DIRBIAS_OUT_SCHEDULE = [
#     ("06:00","09:00", 1),    # dampen center -> periph AM
#     ("15:00","21:00", 10),   # boost evening egress
# ]

# Legacy compat
DIRBIAS_FIXED = (10.0, 1.0)
DIRBIAS_PHASE_MAP = {
    "AM":     (1.50, 0.70),
    "MIDDAY": (1.00, 1.00),
    "PM":     (0.70, 1.50),
    "ELSE":   (1.00, 1.00),
}
AM_HOURS_SET     = {6, 7, 8, 9, 10}
MIDDAY_HOURS_SET = {11, 12, 13}
PM_HOURS_SET     = {14, 15, 16, 17, 18, 19, 20, 21}

# =========================
# Per-bin population control
# =========================
TOTAL_PER_BIN_MODE  = "fixed"                   # {"fixed","template_mean","template_mean_day0"}
TOTAL_PER_BIN_FIXED = 12000

# =========================
# Pairwise distance envelopes
# =========================
ENABLE_PAIRWISE_DISTANCE_ENVELOPES = True
SELF_DISTANCE_MODE = "double_diag"  # {"single_diag","double_diag"}
DUMP_PAIRWISE_ENVELOPES_JSON = False  # Optional debug dump

# =========================
# Diagnostics toggles
# =========================
ENABLE_DIAG_MODEL_EMP_COMPARE = False
ENABLE_DIAG_TEMPORAL          = True
ENABLE_DIAG_MEAN_BIN          = True
ENABLE_DIAG_DAILY             = True
WRITE_DIRBIAS_PREVIEW_CSV     = True  # quick sanity CSV for scheduled bias

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
# Time helpers & phases
# =========================
def build_time_bins_for_window(d_start: str, d_end: str, hh_start: int, hh_end: int, res_min: int) -> List[pd.Timestamp]:
    ds = pd.to_datetime(d_start); de = pd.to_datetime(d_end)
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
    if not H: return None, None
    return int(min(H)), int(max(H))

def phase_name_of_minute(minute_of_day: int) -> str:
    h = minute_of_day // 60
    if h in AM_HOURS_SET: return "AM"
    if h in MIDDAY_HOURS_SET: return "MIDDAY"
    if h in PM_HOURS_SET: return "PM"
    return "ELSE"

def phase_s_of_minute(minute_of_day: int) -> float:
    name = phase_name_of_minute(minute_of_day)
    if name == "AM": return +1.0
    if name == "PM": return -1.0
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
                if bit: lon_interval[0] = mid
                else:   lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit: lat_interval[0] = mid
                else:   lat_interval[1] = mid
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
    if not USE_REGION_FILTER or not region_name: return df
    if region_name not in REGION_BBOXES:
        warn(f"Region '{region_name}' not in REGION_BBOXES; skipping region filter.")
        return df

    bbox = REGION_BBOXES[region_name]
    buf_km = BUFFER_KM_BY_COUNTRY.get(COUNTRY, 0.0)
    bbox_b = _buffer_bbox(*bbox, buffer_km=buf_km)

    uniq = pd.unique(pd.concat([df[START_COL], df[END_COL]], ignore_index=True))
    gh2pt = {}
    for g in uniq:
        try:
            lat, lon = geohash_decode(str(g))
            gh2pt[str(g)] = (lon, lat)  # lon,lat for bbox
        except Exception:
            gh2pt[str(g)] = (None, None)

    def in_region(g: str) -> bool:
        lonlat = gh2pt.get(str(g), (None, None))
        if lonlat[0] is None: return False
        return _point_in_bbox(lonlat[0], lonlat[1], bbox_b)

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
# Center selection helpers
# =========================
def _edge_km_from_precision(p: int) -> float:
    table = {4:19.5, 5:4.9, 6:1.2, 7:0.153, 8:0.038}
    return float(table.get(int(p), 4.9))

CELL_EDGE_KM_EST   = _edge_km_from_precision(GEOHASH_PRECISION)
CELL_DIAG_KM_EST   = CELL_EDGE_KM_EST * math.sqrt(2.0)

def _equirect_km_offsets(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    phi = math.radians((lat0 + lat1) * 0.5)
    kx = 111.320 * math.cos(phi)
    ky = 110.574
    dx_km = (lon1 - lon0) * kx
    dy_km = (lat1 - lat0) * ky
    return float(dx_km), float(dy_km)

def infer_centers_by_inflow(df: pd.DataFrame, k: int) -> List[str]:
    if COUNT_COL not in df.columns:
        s = df[END_COL].value_counts()
    else:
        s = df.groupby(END_COL)[COUNT_COL].sum().sort_values(ascending=False)
    return list(s.head(k).index.astype(str))

def _centroid_of_inflow(df: pd.DataFrame) -> Tuple[float,float]:
    # Weighted by inbound counts (end nodes)
    if COUNT_COL not in df.columns:
        s = df[END_COL].value_counts()
    else:
        s = df.groupby(END_COL)[COUNT_COL].sum()
    if s.empty:
        raise ValueError("Cannot compute inflow centroid: no counts.")
    totals = s.sum()
    lat_sum = lon_sum = 0.0
    for gh, w in s.items():
        lat, lon = geohash_decode(str(gh))
        lat_sum += lat * float(w)
        lon_sum += lon * float(w)
    return float(lat_sum/totals), float(lon_sum/totals)

def _centroid_of_all_nodes(nodes: Iterable[str]) -> Tuple[float,float]:
    nodes = list(nodes)
    if not nodes:
        raise ValueError("No nodes for centroid.")
    lat_sum = lon_sum = 0.0
    for gh in nodes:
        lat, lon = geohash_decode(str(gh))
        lat_sum += lat; lon_sum += lon
    return float(lat_sum/len(nodes)), float(lon_sum/len(nodes))

def _center_reference_latlon(
    nodes: List[str],
    dfw: pd.DataFrame,
    mode: str,
    custom_latlon: Tuple[float,float]
) -> Tuple[float,float]:
    mode = (mode or "inflow_centroid").lower()
    if mode == "inflow_centroid":
        return _centroid_of_inflow(dfw)
    if mode == "all_nodes_centroid":
        return _centroid_of_all_nodes(nodes)
    if mode == "custom":
        return float(custom_latlon[0]), float(custom_latlon[1])
    warn(f"Unknown CENTER_REFERENCE='{mode}', falling back to inflow_centroid.")
    return _centroid_of_inflow(dfw)

def _rank_by_radius(nodes: List[str], ref_lat: float, ref_lon: float, k: int) -> List[str]:
    dists = []
    for gh in nodes:
        lat, lon = geohash_decode(gh)
        d = haversine_km(ref_lat, ref_lon, lat, lon)
        dists.append((gh, d))
    dists.sort(key=lambda t: t[1])
    return [g for g,_ in dists[:max(1,int(k))]]

def _radius_threshold(nodes: List[str], ref_lat: float, ref_lon: float, r_km: float) -> List[str]:
    out = []
    for gh in nodes:
        lat, lon = geohash_decode(gh)
        d = haversine_km(ref_lat, ref_lon, lat, lon)
        if d <= float(r_km):
            out.append(gh)
    if not out:
        warn("radius_threshold produced 0 center tiles; falling back to closest 1.")
        return _rank_by_radius(nodes, ref_lat, ref_lon, 1)
    return out

def select_centers(
    nodes: List[str],
    dfw: pd.DataFrame,
    mode: str,
    reference_mode: str,
    custom_latlon: Tuple[float,float],
    radius_km: float,
    topk: int,
    fixed_ids: List[str],
    file_path: str,
) -> List[str]:
    mode = (mode or "inflow_topk").lower()
    reference_mode = (reference_mode or "inflow_centroid").lower()

    if mode == "inflow_topk":
        centers = infer_centers_by_inflow(dfw, max(1,int(topk)))
        return centers

    ref_lat, ref_lon = _center_reference_latlon(nodes, dfw, reference_mode, custom_latlon)

    if mode == "rank_by_radius":
        return _rank_by_radius(nodes, ref_lat, ref_lon, max(1,int(topk)))
    if mode == "radius_threshold":
        return _radius_threshold(nodes, ref_lat, ref_lon, float(radius_km))
    if mode == "fixed_ids":
        if not fixed_ids:
            warn("CENTER_FIXED_IDS empty; falling back to inflow_topk.")
            return infer_centers_by_inflow(dfw, max(1,int(topk)))
        ok = [g for g in fixed_ids if g in set(nodes)]
        if not ok:
            warn("None of CENTER_FIXED_IDS found in nodes; falling back to inflow_topk.")
            return infer_centers_by_inflow(dfw, max(1,int(topk)))
        return ok
    if mode == "file":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ids = [ln.strip() for ln in f if ln.strip()]
            ok = [g for g in ids if g in set(nodes)]
            if not ok:
                warn("CENTER file provided but none matched nodes; falling back to inflow_topk.")
                return infer_centers_by_inflow(dfw, max(1,int(topk)))
            return ok
        except Exception as e:
            warn(f"Could not read CENTER_FILE_PATH='{file_path}': {e}; falling back to inflow_topk.")
            return infer_centers_by_inflow(dfw, max(1,int(topk)))

    warn(f"Unknown CENTER_SELECTION_MODE='{mode}', using inflow_topk.")
    return infer_centers_by_inflow(dfw, max(1,int(topk)))

# =========================
# Neighbor graph (lattice/knn)
# =========================
def _lattice_neighbor_indices(nodes: List[str], K_cap: int, max_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    N = len(nodes)
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    latlon = np.array([geohash_decode(g) for g in nodes], dtype=float)
    lat_ref = float(latlon[:, 0].mean()); lon_ref = float(latlon[:, 1].mean())

    def _to_xy_km(lat: float, lon: float) -> Tuple[float, float]:
        return _equirect_km_offsets(lat_ref, lon_ref, lat, lon)

    centers_xy = np.array([_to_xy_km(float(lat), float(lon)) for lat, lon in latlon], dtype=float)

    e_km = float(CELL_EDGE_KM_EST)
    gx = np.round(centers_xy[:, 0] / e_km).astype(int)
    gy = np.round(centers_xy[:, 1] / e_km).astype(int)

    grid = {(int(gx[i]), int(gy[i])): i for i in range(N)}

    offsets: List[Tuple[int,int,int,float]] = []  # (dx, dy, cheb, euclid)
    for dx in range(-max_rings, max_rings + 1):
        for dy in range(-max_rings, max_rings + 1):
            cheb = max(abs(dx), abs(dy)); eu = math.hypot(dx, dy)
            offsets.append((dx, dy, cheb, eu))
    offsets.sort(key=lambda t: (t[2], t[3]))

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
            lst_idx.append(j)
            d_km = haversine_km(float(latlon[i, 0]), float(latlon[i, 1]), float(latlon[j, 0]), float(latlon[j, 1]))
            lst_dst.append(float(d_km))
        if not lst_idx or lst_idx[0] != i:
            if i in lst_idx:
                pos = lst_idx.index(i)
                lst_idx.pop(pos); lst_dst.pop(pos)
            lst_idx.insert(0, i); lst_dst.insert(0, 0.0)
        if K_cap is not None and K_cap > 0:
            lst_idx = lst_idx[:int(K_cap)]; lst_dst = lst_dst[:int(K_cap)]
        neigh_lists.append(lst_idx); dist_lists.append(lst_dst)

    K_eff = max(len(x) for x in neigh_lists)
    nb_idx  = np.full((N, K_eff), -1, dtype=int)
    nb_dist = np.zeros((N, K_eff), dtype=float)

    for i in range(N):
        row_idx = neigh_lists[i]; row_dst = dist_lists[i]
        if len(row_idx) < K_eff:
            pad_len = K_eff - len(row_idx)
            row_idx = row_idx + [i] * pad_len
            row_dst = row_dst + [0.0] * pad_len
        nb_idx[i, :]  = np.array(row_idx, dtype=int)
        nb_dist[i, :] = np.array(row_dst, dtype=float)

    return nb_idx, nb_dist

def _knn_neighbor_indices(nodes: List[str], K: int) -> Tuple[np.ndarray, np.ndarray]:
    N = len(nodes)
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)
    K_eff = max(1, min(int(K), N))
    latlons = np.array([geohash_decode(g) for g in nodes], dtype=float)
    dmat = np.zeros((N, N), dtype=float)
    for j in range(N):
        latj, lonj = latlons[j]
        for i in range(j + 1, N):
            lati, loni = latlons[i]
            d = haversine_km(latj, lonj, lati, loni)
            dmat[j, i] = d; dmat[i, j] = d
    nb_idx = np.argpartition(dmat, kth=K_eff - 1, axis=0)[:K_eff, :].T
    nb_dist = np.take_along_axis(dmat, nb_idx, axis=1)
    for j in range(N):
        row_idx = nb_idx[j]; row_dist = nb_dist[j]
        if j not in row_idx:
            far = int(np.argmax(row_dist)); row_idx[far] = j; row_dist[far] = 0.0
        order = np.argsort(row_dist)
        nb_idx[j] = row_idx[order]; nb_dist[j] = row_dist[order]
    # Optional cutoff
    if NEIGHBOR_DISTANCE_CUTOFF_KM is not None:
        cutoff_km = float(NEIGHBOR_DISTANCE_CUTOFF_KM)
        N, K_eff = nb_idx.shape
        for j in range(N):
            keep = (nb_idx[j] == j) | (nb_dist[j] <= cutoff_km)
            if np.all(keep): continue
            row_idx = nb_idx[j][keep]
            row_dst = nb_dist[j][keep]
            pad = K_eff - row_idx.size
            if pad > 0:
                row_idx = np.concatenate([row_idx, np.full(pad, j, dtype=int)])
                row_dst = np.concatenate([row_dst, np.zeros(pad, dtype=float)])
            nb_idx[j]  = row_idx[:K_eff]
            nb_dist[j] = row_dst[:K_eff]
    return nb_idx.astype(int), nb_dist.astype(float)

def neighbor_indices(nodes: List[str], K: int, scheme: str, max_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    scheme = (scheme or "knn").lower()
    if scheme == "lattice":
        return _lattice_neighbor_indices(nodes, K_cap=int(K), max_rings=int(max_rings))
    return _knn_neighbor_indices(nodes, int(K))

# =========================
# Schedules & painter helpers (with HOLD semantics)
# =========================
def _parse_hhmm(hhmm: str) -> int:
    hh, mm = hhmm.split(":"); return int(hh) * 60 + int(mm)

def _minute_to_bin(minute_of_day: int) -> int:
    return int((minute_of_day - WINDOW_START_HH * 60) // TIME_RES_MIN)

def _segment_bins(lo_min: int, hi_min: int, bins_per_day: int) -> List[int]:
    """
    Convert a [start, end) minute range into covered bin indices (end-exclusive).
    Handles wrap across midnight. Interprets end="00:00" as 24:00 if start>end.
    """
    if bins_per_day <= 0:
        return []

    wrap = lo_min > hi_min
    if wrap and hi_min == 0:
        hi_for_tail = 24 * 60
    else:
        hi_for_tail = hi_min

    def end_excl_to_last_bin(end_min: int) -> int:
        last_min = max(0, end_min - 1)
        return max(0, min(bins_per_day - 1, _minute_to_bin(last_min)))

    lo_bin = max(0, min(bins_per_day - 1, _minute_to_bin(lo_min)))

    if not wrap:
        hi_bin = end_excl_to_last_bin(hi_for_tail)
        if hi_bin < lo_bin:
            return []
        return list(range(lo_bin, hi_bin + 1))

    hi_bin_tail = end_excl_to_last_bin(hi_for_tail)
    bins = list(range(lo_bin, bins_per_day))
    if hi_min > 0:
        hi_bin_head = end_excl_to_last_bin(hi_min)
        bins += list(range(0, hi_bin_head + 1))
    return bins

def _ramp_fill(arr: np.ndarray, bins: List[int], start_val: float, end_val: float) -> None:
    """
    Linearly ramp arr over 'bins' from start_val at bins[0] to end_val at bins[-1].
    One-bin span becomes an immediate set to end_val.
    """
    if not bins:
        return
    if len(bins) == 1:
        arr[bins[0]] = float(end_val)
        return
    seg = np.linspace(float(start_val), float(end_val), num=len(bins))
    for k, b in enumerate(bins):
        arr[b] = float(seg[k])

def _build_array_with_hold(
    base_val: float,
    schedule: List[Tuple[str,str,Union[float,Tuple[float,float]]]],
    mode: str,
    bins_per_day: int,
    clip01: bool
) -> np.ndarray:
    """
    Generic painter with HOLD semantics.
    mode: "absolute" | "multiplier"
    - absolute: seed to base_val; scalar target c ramps current→c; tuple (v0,v1) ramps v0→v1
    - multiplier: seed factor=1; scalar c ramps 1→c; tuple (c0,c1) ramps c0→c1; final=base*factor
    clip01: if True, clip to [0,1] at end (useful for stay arrays)
    """
    mode = (mode or "absolute").lower()
    if mode == "absolute":
        arr = np.full(bins_per_day, float(base_val), dtype=float)
        painted = np.zeros(bins_per_day, dtype=bool)
        for (t0s, t1s, tgt) in (schedule or []):
            lo_min, hi_min = _parse_hhmm(t0s), _parse_hhmm(t1s)
            bins = _segment_bins(lo_min, hi_min, bins_per_day)
            if not bins:
                continue
            if isinstance(tgt, tuple):
                _ramp_fill(arr, bins, float(tgt[0]), float(tgt[1]))
            else:
                start_val = float(arr[bins[0]])
                _ramp_fill(arr, bins, start_val, float(tgt))
            painted[bins] = True
        for b in range(1, bins_per_day):
            if not painted[b]:
                arr[b] = arr[b-1]
        return np.clip(arr, 0.0, 1.0) if clip01 else arr

    base_arr = np.full(bins_per_day, float(base_val), dtype=float)
    fac = np.ones(bins_per_day, dtype=float)
    painted = np.zeros(bins_per_day, dtype=bool)
    for (t0s, t1s, tgt) in (schedule or []):
        lo_min, hi_min = _parse_hhmm(t0s), _parse_hhmm(t1s)
        bins = _segment_bins(lo_min, hi_min, bins_per_day)
        if not bins:
            continue
        if isinstance(tgt, tuple):
            seg = np.linspace(float(tgt[0]), float(tgt[1]), len(bins))
        else:
            seg = np.linspace(1.0, float(tgt), len(bins))
        for k, b in enumerate(bins):
            fac[b] *= float(seg[k])
        painted[bins] = True
    for b in range(1, bins_per_day):
        if not painted[b]:
            fac[b] = fac[b-1]
    out = base_arr * fac
    return np.clip(out, 0.0, 1.0) if clip01 else out

def _build_dirbias_arrays(
    base_in: float,
    base_out: float,
    sched_in: List[Tuple[str,str,Union[float,Tuple[float,float]]]],
    sched_out: List[Tuple[str,str,Union[float,Tuple[float,float]]]],
    mode: str,
    bins_per_day: int
) -> Tuple[np.ndarray, np.ndarray]:
    kin  = _build_array_with_hold(base_in,  sched_in,  mode, bins_per_day, clip01=False)
    kout = _build_array_with_hold(base_out, sched_out, mode, bins_per_day, clip01=False)
    return kin, kout

# =========================
# Random-bucket helper utilities
# =========================
def _validate_bucket_spec(spec: Dict[int,int]) -> None:
    if not spec:
        raise ValueError("Bucket spec is empty.")
    if any((k <= 0 or v <= 0) for k, v in spec.items()):
        raise ValueError("Bucket spec keys/values must be positive and non-zero.")
    if sum(spec.keys()) != 100:
        raise ValueError(f"Bucket spec tile percentages must sum to 100 (got {sum(spec.keys())}).")
    if sum(spec.values()) != 100:
        raise ValueError(f"Bucket spec quantity percentages must sum to 100 (got {sum(spec.values())}).")

def _largest_remainder_counts(n: int, percents: List[int]) -> List[int]:
    exact = [p * n / 100.0 for p in percents]
    floors = [int(math.floor(x)) for x in exact]
    rem = n - sum(floors)
    if rem > 0:
        fracs = [(exact[i] - floors[i], i) for i in range(len(percents))]
        fracs.sort(reverse=True)
        for _, idx in fracs[:rem]:
            floors[idx] += 1
    return floors

def _assign_tiles_to_buckets(tile_indices: np.ndarray, bucket_counts: List[int], rng: np.random.Generator) -> List[np.ndarray]:
    n = tile_indices.size
    if sum(bucket_counts) != n:
        raise ValueError("Bucket counts must sum to number of tiles.")
    rng.shuffle(tile_indices)
    out = []
    start = 0
    for c in bucket_counts:
        out.append(tile_indices[start:start+c])
        start += c
    return out

def _mass_factors_for_group(n_group: int, spec: Dict[int,int], rng: np.random.Generator, group_indices: np.ndarray) -> np.ndarray:
    """
    Produce per-tile factors whose average across the group is 1.0.
    For a bucket with q% of quantity assigned to n_b tiles, factor = (q/100 * n_group) / n_b.
    """
    _validate_bucket_spec(spec)
    tile_perc = list(spec.keys())
    qty_perc  = list(spec.values())
    if len(tile_perc) > n_group:
        raise ValueError(f"Bucket count ({len(tile_perc)}) exceeds number of tiles in group ({n_group}).")

    bucket_tile_counts = _largest_remainder_counts(n_group, tile_perc)
    if any(c == 0 for c in bucket_tile_counts):
        raise ValueError("A bucket would receive 0 tiles; refine tile percentages for the group size.")

    # Assign tiles to buckets
    buckets = _assign_tiles_to_buckets(group_indices.copy(), bucket_tile_counts, rng)

    # Build factor array
    factors = np.ones(n_group, dtype=float)
    for b_idx, tiles in enumerate(buckets):
        n_b = tiles.size
        q_b = qty_perc[b_idx] / 100.0
        f_b = (q_b * n_group) / float(n_b)  # guarantees group mean of 1.0
        factors_indices = np.isin(group_indices, tiles)
        factors[factors_indices] = f_b
    return factors

def _x_shares_for_group(total_share: float, n_group: int, spec: Dict[int,int], rng: np.random.Generator, group_indices: np.ndarray) -> np.ndarray:
    """
    Produce per-tile absolute shares for x0 that sum to total_share (group total).
    Within each bucket, equal per tile: share = total_share * (q_b/100) / n_b.
    """
    _validate_bucket_spec(spec)
    tile_perc = list(spec.keys())
    qty_perc  = list(spec.values())
    if len(tile_perc) > n_group:
        raise ValueError(f"Bucket count ({len(tile_perc)}) exceeds number of tiles in group ({n_group}).")

    bucket_tile_counts = _largest_remainder_counts(n_group, tile_perc)
    if any(c == 0 for c in bucket_tile_counts):
        raise ValueError("A bucket would receive 0 tiles; refine tile percentages for the group size.")

    buckets = _assign_tiles_to_buckets(group_indices.copy(), bucket_tile_counts, rng)

    shares = np.zeros(n_group, dtype=float)
    for b_idx, tiles in enumerate(buckets):
        n_b = tiles.size
        q_b = qty_perc[b_idx] / 100.0
        per_tile = (total_share * q_b) / float(n_b)
        mask = np.isin(group_indices, tiles)
        shares[mask] = per_tile
    return shares

# =========================
# Mass & X builders (no radial)
# =========================
def assign_center_periph_values_for_nodes(nodes: List[str], centers: List[str], center_val: float, periph_val: float) -> np.ndarray:
    centers_set = set(centers)
    vals = np.full(len(nodes), float(periph_val), dtype=float)
    for idx, g in enumerate(nodes):
        if g in centers_set:
            vals[idx] = float(center_val)
    return vals

def _template_inflow_vector(dfw: pd.DataFrame, nodes: List[str], mode: str, hour_set: Optional[set[int]] = None) -> np.ndarray:
    if COUNT_COL not in dfw.columns or TIME_COL not in dfw.columns:
        return np.ones(len(nodes), dtype=float)
    t = pd.to_datetime(dfw[TIME_COL], errors="coerce", utc=False)
    sel = dfw.copy(); sel["_t0"] = t
    if mode == "day0":
        d0 = t.dt.floor('D').min(); sel = sel[t.dt.floor('D') == d0]
    if mode == "window" and hour_set:
        sel = sel[sel["_t0"].dt.hour.isin(list(hour_set))]
    s = sel.groupby(END_COL)[COUNT_COL].sum()
    v = np.array([float(s.get(g, 0.0)) for g in nodes], dtype=float)
    v = np.where(v <= 0, 1.0, v)
    return v

def build_initial_masses(dfw: pd.DataFrame, nodes: List[str], centers: List[str]) -> np.ndarray:
    mode = str(MASS_INIT_MODE).lower()
    if mode == "flat":
        M = np.ones(len(nodes), dtype=float)
    elif mode == "template_inflow_day0":
        M = _template_inflow_vector(dfw, nodes, mode="day0")
    elif mode == "template_window_mean":
        M = _template_inflow_vector(dfw, nodes, mode="window", hour_set=None)
    elif mode == "center_periph":
        M = assign_center_periph_values_for_nodes(nodes, centers, MASS_CENTER_PERIPH[0], MASS_CENTER_PERIPH[1])
    else:
        warn(f"Unknown MASS_INIT_MODE={MASS_INIT_MODE}; falling back to flat")
        M = np.ones(len(nodes), dtype=float)
    M = M / (M.mean() + 1e-12)
    return np.maximum(M, 0.0)

def power_stationary(P: np.ndarray, tol=1e-12, maxit=10000) -> np.ndarray:
    N = P.shape[0]
    x = np.full(N, 1.0/N, dtype=float)
    for _ in range(maxit):
        x_new = P @ x
        s = x_new.sum(); x_new = x_new / s if s > 0 else np.full(N, 1.0/N, dtype=float)
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new
        x = x_new
    return x

def periodic_fixed_point(P_daily: List[np.ndarray], tol: float = 1e-12, max_days: int = 2000) -> np.ndarray:
    if not P_daily:
        raise ValueError("P_daily empty")
    N = P_daily[0].shape[0]
    x = np.full(N, 1.0/N, dtype=float)
    for _ in range(max_days):
        x_prev = x.copy()
        for Pb in P_daily:
            x = Pb @ x
            s = x.sum(); x = x / s if s > 0 else np.full(N, 1.0/N, dtype=float)
        if np.linalg.norm(x - x_prev, 1) < tol:
            return x
    return x

def compute_initial_state(P_daily: List[np.ndarray], nodes: List[str], dfw: pd.DataFrame, centers: List[str]) -> np.ndarray:
    mode = str(INIT_X_MODE).lower()
    N = len(nodes)
    if mode == "flat":
        return np.full(N, 1.0/N, dtype=float)
    if mode == "template_inflow_day0":
        v = _template_inflow_vector(dfw, nodes, mode="day0"); v_sum = v.sum()
        return (v / v_sum) if v_sum > 0 else np.full(N, 1.0/N)
    if mode == "template_window_mean":
        v = _template_inflow_vector(dfw, nodes, mode="window", hour_set=None); v_sum = v.sum()
        return (v / v_sum) if v_sum > 0 else np.full(N, 1.0/N)
    if mode == "stationary_meanp":
        Pm = np.mean(np.stack(P_daily, axis=0), axis=0)
        x0 = power_stationary(Pm)
        return x0 / (x0.sum() + 1e-12)
    if mode == "periodic_fixed_point":
        x0 = periodic_fixed_point(P_daily)
        return x0 / (x0.sum() + 1e-12)
    if mode == "center_periph":
        x0 = assign_center_periph_values_for_nodes(nodes, centers, X_CENTER_PERIPH[0], X_CENTER_PERIPH[1])
        x0 = x0 / (x0.sum() + 1e-12)
        return x0
    Pm = np.mean(np.stack(P_daily, axis=0), axis=0)
    x0 = power_stationary(Pm)
    return x0 / (x0.sum() + 1e-12)

# =========================
# Directional bias arrays
# =========================
def _phase_map_to_arrays(bins_per_day: int) -> Tuple[np.ndarray,np.ndarray]:
    arr_in  = np.ones(bins_per_day, dtype=float)
    arr_out = np.ones(bins_per_day, dtype=float)
    for b in range(bins_per_day):
        minute = WINDOW_START_HH*60 + b*TIME_RES_MIN
        ph = phase_name_of_minute(minute)
        kin, kout = DIRBIAS_PHASE_MAP.get(ph, DIRBIAS_PHASE_MAP.get("ELSE",(1.0,1.0)))
        arr_in[b]  = float(kin)
        arr_out[b] = float(kout)
    return arr_in, arr_out

def _dirbias_for_bin(b: int, k_in_arr: np.ndarray, k_out_arr: np.ndarray) -> Tuple[float,float]:
    return float(k_in_arr[int(b)]), float(k_out_arr[int(b)])

# =========================
# Markov builder with directional bias
# =========================
def _stay_prob_for_origin(is_center_origin: bool, minute_of_day: int, arr_c: np.ndarray, arr_p: np.ndarray) -> float:
    bins_per_day = arr_c.size
    b = int(_minute_to_bin(minute_of_day)) % max(1, bins_per_day)
    base = float(arr_c[b] if is_center_origin else arr_p[b])
    if float(LAMBDA_STAY) != 0.0:
        s = phase_s_of_minute(minute_of_day)
        sign = +1.0 if is_center_origin else -1.0
        base = base + float(LAMBDA_STAY) * s * sign
    return float(min(1.0, max(0.0, base)))

def build_P_for_bin(
    nodes: List[str],
    centers_set: set,
    bin_index: int,
    minute_of_day: int,
    M_base_vec: np.ndarray,
    nb_idx: np.ndarray,
    nb_dist: np.ndarray,
    arr_stay_center: np.ndarray,
    arr_stay_periph: np.ndarray,
    k_in_arr: np.ndarray,
    k_out_arr: np.ndarray,
) -> np.ndarray:
    N, K = nb_idx.shape

    Mw = np.power(np.maximum(M_base_vec, 0.0), ALPHA)
    is_center = np.array([g in centers_set for g in nodes], dtype=bool)
    k_in, k_out = _dirbias_for_bin(bin_index, k_in_arr, k_out_arr)

    P = np.zeros((N, N), dtype=float)
    for j in range(N):
        pst = _stay_prob_for_origin(bool(is_center[j]), minute_of_day, arr_stay_center, arr_stay_periph)
        neigh = nb_idx[j]
        distk = nb_dist[j]

        if DIST_DECAY_MODE == "power":
            denom = np.power(np.maximum(distk, 1e-12), BETA)
            w = Mw[neigh] / denom
        else:
            w = Mw[neigh] * np.exp(-GAMMA_DECAY * distk)

        if NEIGHBOR_INNER_K > 0 and NEIGHBOR_BOOST not in (None, 1.0, 0.0):
            inner_mask = np.zeros_like(neigh, dtype=bool)
            inner_mask[1:min(NEIGHBOR_INNER_K, len(inner_mask))] = True
            w = w * np.where(inner_mask & (neigh != j), float(NEIGHBOR_BOOST), 1.0)

        mask = (neigh != j)
        neigh_noself = neigh[mask]
        w_noself = w[mask]
        if ENABLE_DIRECTIONAL_BIAS:
            origin_center = bool(is_center[j])
            mult = np.ones_like(w_noself)
            for idx_r, i in enumerate(neigh_noself):
                dest_center = bool(is_center[int(i)])
                if (not origin_center) and dest_center:
                    mult[idx_r] = float(k_in)
                elif origin_center and (not dest_center):
                    mult[idx_r] = float(k_out)
                else:
                    mult[idx_r] = 1.0
            w_noself = w_noself * mult

        ssum = float(np.sum(w_noself))
        if ssum <= 0.0:
            P[j, j] = 1.0
            continue
        w_noself = w_noself / ssum

        P[j, j] = pst
        spread = 1.0 - pst
        P[neigh_noself, j] += spread * w_noself

        colsum = P[:, j].sum()
        if colsum > 0:
            P[:, j] /= colsum
        else:
            P[j, j] = 1.0

    return P

# =========================
# Transitions persistence (pattern + manifest)
# =========================
def run_dir_name() -> str:
    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y-%m-%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y-%m-%d")
    nb_tag = f"nb-{NEIGHBOR_SCHEME}K{int(NEIGHBOR_K)}"
    if not ENABLE_DIRECTIONAL_BIAS:
        dirbias_tag = "dirbias-off"
    elif DIRBIAS_MODE == "scheduled":
        dirbias_tag = f"dirbias-sched-{DIRBIAS_SCHEDULE_MODE}"
    elif DIRBIAS_MODE == "fixed":
        ki, ko = DIRBIAS_FIXED
        dirbias_tag = f"dirbias-fixed{float(ki):.2f}-{float(ko):.2f}"
    else:
        dirbias_tag = "dirbias-legacyphased"
    return f"{start_tag}_{end_tag}_m{TIME_RES_MIN}_x-{INIT_X_MODE}_m-{MASS_INIT_MODE}_{nb_tag}_{dirbias_tag}"

def ensure_run_dirs() -> Dict[str, str]:
    name = run_dir_name()
    rd = os.path.join(RUNS_DIR, name)
    subplots = os.path.join(rd, "plots")
    transitions = os.path.join(rd, "transitions")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(subplots, exist_ok=True)
    os.makedirs(transitions, exist_ok=True)
    return {"run": rd, "plots": subplots, "transitions": transitions}

def transitions_paths(run_paths: Dict[str,str]) -> Tuple[str,str]:
    npy = os.path.join(run_paths["transitions"], f"pep_P_daily_pattern_m{TIME_RES_MIN}.npy")
    man = os.path.join(run_paths["transitions"], f"pep_P_daily_pattern_m{TIME_RES_MIN}.manifest.json")
    return npy, man

def write_transitions(P_daily: List[np.ndarray], nodes: List[str], manifest_path: str, npy_path: str, extra_meta: Optional[dict] = None) -> None:
    if not SAVE_TRANSITIONS:
        return
    arr = np.stack(P_daily, axis=0)
    np.save(npy_path, arr)
    meta = {
        "nodes": nodes,
        "bins_per_day": int(arr.shape[0]),
        "alpha": float(ALPHA),
        "beta": float(BETA),
        "dist_decay_mode": DIST_DECAY_MODE,
        "gamma_decay": float(GAMMA_DECAY),
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
        # directional bias
        "enable_directional_bias": bool(ENABLE_DIRECTIONAL_BIAS),
        "dirbias_mode": DIRBIAS_MODE,
        "dirbias_schedule_mode": DIRBIAS_SCHEDULE_MODE,
        "dirbias_base": DIRBIAS_BASE,
        "dirbias_in_schedule": DIRBIAS_IN_SCHEDULE,
        "dirbias_out_schedule": DIRBIAS_OUT_SCHEDULE,
        "dirbias_fixed": DIRBIAS_FIXED,
        "dirbias_phase_map": DIRBIAS_PHASE_MAP,
        # stay & mass
        "lambda_stay": float(LAMBDA_STAY),
        "enable_stay_schedule": bool(ENABLE_STAY_BASE_SCHEDULE),
        "stay_schedule_mode": STAY_SCHEDULE_MODE,
        "p_stay_base_mode": P_STAY_BASE_MODE,
        "center_stay_segments": CENTER_STAY_SCHEDULE,
        "periph_stay_segments": PERIPH_STAY_SCHEDULE,
        "enable_mass_schedule": bool(ENABLE_MASS_BASE_SCHEDULE),
        "mass_schedule_mode": MASS_SCHEDULE_MODE,
        "mass_time_mode": MASS_TIME_MODE,
        "mass_base_mode": MASS_BASE_MODE,
        "center_mass_segments": CENTER_MASS_SCHEDULE,
        "periph_mass_segments": PERIPH_MASS_SCHEDULE,
        # random buckets
        "mass_assignment_mode": MASS_ASSIGNMENT_MODE,
        "mass_buckets_center": MASS_BUCKETS_CENTER,
        "mass_buckets_periph": MASS_BUCKETS_PERIPH,
        "mass_random_seed": MASS_RANDOM_SEED,
        "x_assignment_mode": X_ASSIGNMENT_MODE,
        "x_buckets_center": X_BUCKETS_CENTER,
        "x_buckets_periph": X_BUCKETS_PERIPH,
        "x_random_seed": X_RANDOM_SEED,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# =========================
# Diagnostics writers (center-only partitions)
# =========================
def _ensure_t0(df: pd.DataFrame) -> pd.DataFrame:
    if "_t0" in df.columns:
        return df
    if TIME_COL not in df.columns:
        raise ValueError("OD frame missing time column.")
    out = df.copy(); out["_t0"] = pd.to_datetime(out[TIME_COL], errors="coerce", utc=False)
    return out

def compute_temporal_diagnostics_df(od: pd.DataFrame, centers: set) -> pd.DataFrame:
    tmp = _ensure_t0(od).copy()
    tmp["_day"]     = tmp["_t0"].dt.floor("D")
    tmp["_bin_idx"] = ((tmp["_t0"].dt.hour * 60 + tmp["_t0"].dt.minute) - WINDOW_START_HH * 60) // TIME_RES_MIN
    tmp["_bin_idx"] = tmp["_bin_idx"].astype(int)
    tmp["_bin_lab"] = tmp["_t0"].dt.strftime("%H:%M")

    START, END = START_COL, END_COL
    is_center_orig = tmp[START].isin(centers)
    is_center_dest = tmp[END].isin(centers)

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

    part_sum = (out["inflow_center"] + out["outflow_center"] + out["self_center"] + out["self_periphery"]).astype(float)
    if not np.allclose(part_sum.values, out["total_bin_flow"].values):
        warn("Partition totals do not equal total_bin_flow for some bins (check centers set and COUNT_COL).")
    return out

def write_pep_temporal_diagnostics(od: pd.DataFrame, centers: set, out_path: str) -> None:
    df = compute_temporal_diagnostics_df(od, centers)
    df.to_csv(out_path, index=False)

def write_pep_mean_bin_balance(od: pd.DataFrame, centers: set, out_path: str) -> None:
    df = compute_temporal_diagnostics_df(od, centers)
    cols = [
        "inflow_center","outflow_center","self_center","self_periphery","total_bin_flow","net_out_center",
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
        "inflow_center","outflow_center","self_center","self_periphery","total_bin_flow","net_out_center",
    ]
    daily = (
        df.groupby("_day", as_index=False)[cols]
          .sum()
          .rename(columns={"total_bin_flow": "total_day_flow"})
          .sort_values("_day")
    )
    daily.to_csv(out_path, index=False)

# =========================
# Validation helpers
# =========================
def _read_generated_od(run_paths: Dict[str,str]) -> pd.DataFrame:
    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y%m%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y%m%d")
    tag = f"{start_tag}_{end_tag}_m{TIME_RES_MIN}"
    path = os.path.join(run_paths["run"], f"pep_od_{tag}.csv")
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

def _load_transitions(run_paths: Dict[str,str]) -> Tuple[List[str], np.ndarray, dict]:
    npy, man = transitions_paths(run_paths)
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

# =========================
# Distance envelopes (pairwise min/max)
# =========================
def _cell_halfspan_km() -> float:
    return float(CELL_DIAG_KM_EST * 0.5)

def _pairwise_envelopes(
    nodes: List[str], nb_idx: np.ndarray
) -> Tuple[Dict[int, Dict[int, Tuple[float,float]]], Dict[str, float]]:
    """
    Returns:
      envelopes[j][i] = (d_min_km, d_max_km) for each (origin j -> neighbor i)
      meta: dict with some scalars for sanity
    """
    N, K = nb_idx.shape
    latlons = np.array([geohash_decode(g) for g in nodes], dtype=float)

    envelopes: Dict[int, Dict[int, Tuple[float,float]]] = {}
    diag = float(CELL_DIAG_KM_EST)

    # Self-hop distance envelope: min is 0, max depends on mode
    if SELF_DISTANCE_MODE == "double_diag":
        self_min = 0.0
        self_max = 2.0 * diag
    else:  # "single_diag"
        self_min = 0.0
        self_max = 1.0 * diag

    for j in range(N):
        envelopes[j] = {}
        latj, lonj = latlons[j]
        for r in range(K):
            i = int(nb_idx[j, r])
            if i == j:
                envelopes[j][i] = (self_min, self_max)
                continue
            lati, loni = latlons[i]
            center_dist = haversine_km(latj, lonj, lati, loni)
            half = _cell_halfspan_km()
            dmin = max(0.0, center_dist - 2*half)
            dmax = center_dist + 2*half
            envelopes[j][i] = (float(dmin), float(dmax))
    meta = {
        "cell_edge_km_est": CELL_EDGE_KM_EST,
        "cell_diag_km_est": CELL_DIAG_KM_EST,
        "self_mode": SELF_DISTANCE_MODE,
    }
    return envelopes, meta

# =========================
# Run: validate
# =========================
def validate():
    banner("PEP VALIDATE")
    run_paths = ensure_run_dirs()

    od = _read_generated_od(run_paths)
    od = od.loc[
        (od["_t0"].dt.floor("D") >= pd.Timestamp(SWEEP_START_DATE)) &
        (od["_t0"].dt.floor("D") <= pd.Timestamp(SWEEP_END_DATE)) &
        (od["_t0"].dt.hour >= WINDOW_START_HH) & (od["_t0"].dt.hour < WINDOW_END_HH)
    ].copy()

    nodes, P_daily, meta = _load_transitions(run_paths)
    centers = set(meta.get("centers", []))
    bins_per_day = int(meta.get("bins_per_day", max(1, int(((WINDOW_END_HH-WINDOW_START_HH)*60)//TIME_RES_MIN))))

    if ENABLE_DIAG_MODEL_EMP_COMPARE:
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
        comp_path = os.path.join(run_paths["run"], "pep_model_empirical_compare.csv")
        comp_df.to_csv(comp_path, index=False)
        info(f"[WRITE] model vs empirical (per-bin, mass-weighted): {comp_path}")
    else:
        info("[SKIP] model vs empirical compare (disabled)")

    start_tag = pd.to_datetime(SWEEP_START_DATE).strftime("%Y%m%d")
    end_tag   = pd.to_datetime(SWEEP_END_DATE).strftime("%Y%m%d")
    tag = f"{start_tag}_{end_tag}_m{TIME_RES_MIN}"

    if ENABLE_DIAG_TEMPORAL:
        td_path = os.path.join(run_paths["run"], f"pep_temporal_diagnostics_{tag}.csv")
        write_pep_temporal_diagnostics(od, centers, td_path)
        info(f"[WRITE] temporal diagnostics: {td_path}")
    else:
        info("[SKIP] temporal diagnostics (disabled)")

    if ENABLE_DIAG_MEAN_BIN:
        mb_path = os.path.join(run_paths["run"], f"pep_mean_bin_balance_{tag}.csv")
        write_pep_mean_bin_balance(od, centers,  mb_path)
        info(f"[WRITE] mean bin balance: {mb_path}")
    else:
        info("[SKIP] mean bin balance (disabled)")

    if ENABLE_DIAG_DAILY:
        dl_path = os.path.join(run_paths["run"], f"pep_daily_balance_{tag}.csv")
        write_pep_daily_balance(od, centers, dl_path)
        info(f"[WRITE] daily balance: {dl_path}")
    else:
        info("[SKIP] daily balance (disabled)")

    info("Validation complete.")

# =========================
# Run: generate
# =========================
def generate():
    banner("PEP GENERATION")
    rng = np.random.default_rng(SEED)
    bins_per_day = int(((WINDOW_END_HH - WINDOW_START_HH) * 60) // TIME_RES_MIN)
    minutes_in_day = [WINDOW_START_HH*60 + b*TIME_RES_MIN for b in range(bins_per_day)]

    # --- Run folder ---
    run_paths = ensure_run_dirs()

    # --- Precompute stay baselines (center/periphery) ---
    if ENABLE_STAY_BASE_SCHEDULE:
        c0, p0 = P_STAY_BASE_MODE
        arr_c = _build_array_with_hold(c0, CENTER_STAY_SCHEDULE, STAY_SCHEDULE_MODE, bins_per_day, clip01=True)
        arr_p = _build_array_with_hold(p0, PERIPH_STAY_SCHEDULE, STAY_SCHEDULE_MODE, bins_per_day, clip01=True)
    else:
        c0, p0 = P_STAY_BASE_MODE
        arr_c = np.full(bins_per_day, float(c0), dtype=float)
        arr_p = np.full(bins_per_day, float(p0), dtype=float)

    # --- Directional bias arrays ---
    if ENABLE_DIRECTIONAL_BIAS:
        if DIRBIAS_MODE == "scheduled":
            k_in_arr, k_out_arr = _build_dirbias_arrays(
                base_in=DIRBIAS_BASE[0],
                base_out=DIRBIAS_BASE[1],
                sched_in=DIRBIAS_IN_SCHEDULE,
                sched_out=DIRBIAS_OUT_SCHEDULE,
                mode=DIRBIAS_SCHEDULE_MODE,
                bins_per_day=bins_per_day
            )
        elif DIRBIAS_MODE == "fixed":
            k_in_arr  = np.full(bins_per_day, float(DIRBIAS_FIXED[0]), dtype=float)
            k_out_arr = np.full(bins_per_day, float(DIRBIAS_FIXED[1]), dtype=float)
        else:
            k_in_arr, k_out_arr = _phase_map_to_arrays(bins_per_day)
    else:
        k_in_arr = k_out_arr = np.ones(bins_per_day, dtype=float)

    if WRITE_DIRBIAS_PREVIEW_CSV:
        prev = pd.DataFrame({
            "bin_idx": np.arange(bins_per_day, dtype=int),
            "bin_label": [f"{(WINDOW_START_HH*60 + b*TIME_RES_MIN)//60:02d}:{(WINDOW_START_HH*60 + b*TIME_RES_MIN)%60:02d}" for b in range(bins_per_day)],
            "k_in": k_in_arr.astype(float),
            "k_out": k_out_arr.astype(float),
        })
        path_prev = os.path.join(run_paths["run"], f"pep_dirbias_schedule_preview_m{TIME_RES_MIN}.csv")
        prev.to_csv(path_prev, index=False)
        kin_min, kin_max = float(k_in_arr.min()), float(k_in_arr.max())
        kout_min, kout_max = float(k_out_arr.min()), float(k_out_arr.max())
        info(f"[WRITE] directional bias preview: {path_prev} (k_in[{kin_min:.3g},{kin_max:.3g}], k_out[{kout_min:.3g},{kout_max:.3g}])")

    # --- Template & region filter ---
    if USE_TEMPLATE and os.path.exists(TEMPLATE_PATH):
        tpl = pd.read_csv(TEMPLATE_PATH)
        dfw = window_slice_template(tpl := tpl)
        dfw = apply_region_filter(dfw, REGION_NAME)
        nodes = sorted(set(dfw[START_COL]).union(dfw[END_COL]))
        if len(nodes) == 0:
            raise ValueError("No nodes discovered from template after region filter.")
        info(f"NODES (unique geohash{GEOHASH_PRECISION}): {len(nodes)}")

        centers = select_centers(
            nodes=nodes,
            dfw=dfw,
            mode=CENTER_SELECTION_MODE,
            reference_mode=CENTER_REFERENCE,
            custom_latlon=CENTER_REF_LATLON,
            radius_km=CENTER_RADIUS_KM,
            topk=CENTER_TOPK,
            fixed_ids=CENTER_FIXED_IDS,
            file_path=CENTER_FILE_PATH,
        )
        info(f"Center tiles (mode={CENTER_SELECTION_MODE}): {centers[:min(10, len(centers))]}{'...' if len(centers)>10 else ''}")
    else:
        raise FileNotFoundError("Template required (USE_TEMPLATE=True).")

    centers_set = set(centers)
    is_center_mask = np.array([g in centers_set for g in nodes], dtype=bool)
    idx_center = np.where(is_center_mask)[0]
    idx_periph = np.where(~is_center_mask)[0]

    # --- Neighbor graph ---
    nb_idx, nb_dist = neighbor_indices(nodes, NEIGHBOR_K, NEIGHBOR_SCHEME, NEIGHBOR_MAX_RINGS)

    # --- Pairwise distance envelopes (optional) ---
    envelopes = None; env_meta = None
    if ENABLE_PAIRWISE_DISTANCE_ENVELOPES:
        envelopes, env_meta = _pairwise_envelopes(nodes, nb_idx)
        if DUMP_PAIRWISE_ENVELOPES_JSON:
            env_path = os.path.join(run_paths["run"], f"pep_envelopes_m{TIME_RES_MIN}.json")
            dump = {
                "meta": env_meta,
                "envelopes": {str(j): {str(i): [float(mn), float(mx)] for i,(mn,mx) in inner.items()} for j, inner in envelopes.items()}
            }
            with open(env_path, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2)
            info(f"[WRITE] pairwise envelopes JSON: {env_path}")

    # --- Mass baselines (scheduled arrays per bin) ---
    M_base_static = build_initial_masses(dfw, nodes, centers)
    if ENABLE_MASS_BASE_SCHEDULE:
        mass_mode = str(MASS_SCHEDULE_MODE or MASS_TIME_MODE)
        center_mass_arr = _build_array_with_hold(MASS_BASE_MODE[0], CENTER_MASS_SCHEDULE, mass_mode, bins_per_day, clip01=False)
        periph_mass_arr = _build_array_with_hold(MASS_BASE_MODE[1], PERIPH_MASS_SCHEDULE, mass_mode, bins_per_day, clip01=False)
    else:
        center_mass_arr = np.full(bins_per_day, float(MASS_BASE_MODE[0]), dtype=float)
        periph_mass_arr = np.full(bins_per_day, float(MASS_BASE_MODE[1]), dtype=float)

    # --- Random-bucket MASS factors (within-group; mean=1) ---
    mass_rng = np.random.default_rng(SEED if MASS_RANDOM_SEED is None else MASS_RANDOM_SEED)
    mass_factors = np.ones(len(nodes), dtype=float)
    if MASS_ASSIGNMENT_MODE == "random_buckets":
        if idx_center.size > 0:
            f_center = _mass_factors_for_group(
                n_group=idx_center.size,
                spec=MASS_BUCKETS_CENTER,
                rng=mass_rng,
                group_indices=idx_center.copy()
            )
            mass_factors[idx_center] = f_center
        if idx_periph.size > 0:
            f_periph = _mass_factors_for_group(
                n_group=idx_periph.size,
                spec=MASS_BUCKETS_PERIPH,
                rng=mass_rng,
                group_indices=idx_periph.copy()
            )
            mass_factors[idx_periph] = f_periph
    elif MASS_ASSIGNMENT_MODE != "uniform":
        warn(f"Unknown MASS_ASSIGNMENT_MODE='{MASS_ASSIGNMENT_MODE}', falling back to 'uniform'.")

    # --- Build P_daily pattern ---
    P_daily = []
    for b, mday in enumerate(minutes_in_day):
        if ENABLE_MASS_BASE_SCHEDULE and (str(MASS_SCHEDULE_MODE).lower() in ("absolute","multiplier")):
            mass_center_bin  = float(center_mass_arr[b])
            mass_periph_bin  = float(periph_mass_arr[b])
            M_base_bin = np.where(
                is_center_mask,
                mass_center_bin,
                mass_periph_bin
            ).astype(float)
        elif ENABLE_MASS_BASE_SCHEDULE and (str(MASS_TIME_MODE).lower() == "absolute"):
            mass_center_bin  = float(_build_array_with_hold(MASS_BASE_MODE[0], CENTER_MASS_SCHEDULE, "absolute", bins_per_day, clip01=False)[b])
            mass_periph_bin  = float(_build_array_with_hold(MASS_BASE_MODE[1], PERIPH_MASS_SCHEDULE, "absolute", bins_per_day, clip01=False)[b])
            M_base_bin = np.where(
                is_center_mask,
                mass_center_bin,
                mass_periph_bin
            ).astype(float)
        else:
            M_base_bin = M_base_static

        # Apply within-group mass factors (group mean preserved at 1.0)
        M_base_bin = M_base_bin * mass_factors

        P_daily.append(
            build_P_for_bin(
                nodes=nodes,
                centers_set=centers_set,
                bin_index=b,
                minute_of_day=mday,
                M_base_vec=M_base_bin,
                nb_idx=nb_idx,
                nb_dist=nb_dist,
                arr_stay_center=arr_c,
                arr_stay_periph=arr_p,
                k_in_arr=k_in_arr,
                k_out_arr=k_out_arr,
            )
        )

    # --- Initial state x0 (between-group from INIT_X_MODE, then within-group bucket redistribution) ---
    x0 = compute_initial_state(P_daily, nodes, dfw, centers)
    x_rng = np.random.default_rng(SEED if X_RANDOM_SEED is None else X_RANDOM_SEED)
    if X_ASSIGNMENT_MODE == "random_buckets":
        total_center = float(x0[idx_center].sum())
        total_periph = float(x0[idx_periph].sum())
        x0_new = np.zeros_like(x0)

        if idx_center.size > 0 and total_center > 0.0:
            shares_center = _x_shares_for_group(
                total_share=total_center,
                n_group=idx_center.size,
                spec=X_BUCKETS_CENTER,
                rng=x_rng,
                group_indices=idx_center.copy()
            )
            x0_new[idx_center] = shares_center

        if idx_periph.size > 0 and total_periph > 0.0:
            shares_periph = _x_shares_for_group(
                total_share=total_periph,
                n_group=idx_periph.size,
                spec=X_BUCKETS_PERIPH,
                rng=x_rng,
                group_indices=idx_periph.copy()
            )
            x0_new[idx_periph] = shares_periph

        # Preserve total mass = 1.0 (should already hold but normalize to be safe)
        s = x0_new.sum()
        if s > 0:
            x0 = x0_new / s
        else:
            warn("X random bucket produced zero total; falling back to original x0.")
    elif X_ASSIGNMENT_MODE != "uniform":
        warn(f"Unknown X_ASSIGNMENT_MODE='{X_ASSIGNMENT_MODE}', falling back to 'uniform'.")

    # --- Persist transitions pattern ---
    trans_npy, manifest = transitions_paths(run_paths)
    write_transitions(
        P_daily, nodes, manifest, trans_npy,
        extra_meta={
            "centers": centers,
            "seed": SEED,
            "mass_init_mode": MASS_INIT_MODE,
            "init_x_mode": INIT_X_MODE,
        },
    )

    # --- Compute per-bin totals from template ---
    template_df = dfw.copy()
    if TIME_COL not in template_df.columns:
        raise RuntimeError("Template is missing TIME_COL; cannot compute per-bin means for TOTAL_PER_BIN_MODE")
    template_df["_t0"] = pd.to_datetime(template_df[TIME_COL], errors="coerce", utc=False)
    template_df = template_df[
        (template_df["_t0"].dt.hour >= WINDOW_START_HH) & (template_df["_t0"].dt.hour <  WINDOW_END_HH)
    ].copy()
    template_df["bin"] = ((template_df["_t0"].dt.hour*60 + template_df["_t0"].dt.minute) - WINDOW_START_HH*60) // TIME_RES_MIN
    template_df["local_date"] = template_df["_t0"].dt.floor("D")

    if COUNT_COL in template_df.columns:
        template_df["_trips"] = template_df[COUNT_COL].astype(float)
    else:
        template_df["_trips"] = 1.0

    template_bin_totals = (
        template_df.groupby(["local_date", "bin"], as_index=False)["_trips"].sum()
        .rename(columns={"_trips": "trip_count"})
    )

    # --- Agent pool ---
    N = len(nodes)
    if TOTAL_PER_BIN_MODE == "fixed":
        total_per_bin = float(TOTAL_PER_BIN_FIXED)
        info(f"[TOTALS] Fixed total per bin: {total_per_bin:.0f}")
    elif TOTAL_PER_BIN_MODE == "template_mean":
        if template_bin_totals.empty:
            raise RuntimeError("[TOTALS] template_bin_totals is empty; cannot compute template_mean")
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

    # deterministic or multinomial assignment from x0
    if DETERMINISTIC_INITIAL_ASSIGNMENT:
        expected = x0 * POOL_SIZE
        counts = np.floor(expected).astype(int)
        remainder = POOL_SIZE - counts.sum()
        if remainder > 0:
            fractional = expected - counts
            top_idx = np.argsort(-fractional)[:remainder]
            counts[top_idx] += 1
        agent_state = np.repeat(np.arange(N), counts)
        rng.shuffle(agent_state)
        initial_counts = counts
    else:
        agent_state = rng.choice(N, size=POOL_SIZE, p=x0, replace=True)
        initial_counts = np.bincount(agent_state, minlength=N)

    initial_agent_state = agent_state.copy()

    out_rows: List[Tuple[str,str,str,int,str,float]] = []

    day = pd.to_datetime(SWEEP_START_DATE)
    end_day = pd.to_datetime(SWEEP_END_DATE)

    while day <= end_day:
        for b, mday in enumerate(minutes_in_day):
            tbin = day + pd.Timedelta(minutes=int(mday)) - pd.Timedelta(minutes=WINDOW_START_HH*60)
            P = P_daily[b]
            print(f"\r[DAY] {day.date()} in progress...", end="", flush=True)

            # --- Markov step: EVERY agent transitions once ---
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

            # --- Emit a row for EVERY agent ---
            for a in range(POOL_SIZE):
                j = int(js_all[a]); i = int(next_state[a])

                # length sampling
                if ENABLE_PAIRWISE_DISTANCE_ENVELOPES and envelopes is not None:
                    env = envelopes.get(j, {}).get(i)
                    if env is None:
                        if i == j:
                            length_km = float(CELL_DIAG_KM_EST if SELF_DISTANCE_MODE == "single_diag" else 2.0*CELL_DIAG_KM_EST)
                        else:
                            if i in nb_idx[j]:
                                rpos = int(np.where(nb_idx[j] == i)[0][0])
                                length_km = float(nb_dist[j, rpos])
                            else:
                                length_km = 0.0
                    else:
                        mn, mx = env
                        u = rng.random()
                        length_km = float(mn + (mx - mn) * u)
                else:
                    if i == j:
                        length_km = float(CELL_DIAG_KM_EST if SELF_DISTANCE_MODE == "single_diag" else 2.0*CELL_DIAG_KM_EST)
                    else:
                        if i in nb_idx[j]:
                            rpos = int(np.where(nb_idx[j] == i)[0][0])
                            length_km = float(nb_dist[j, rpos])
                        else:
                            length_km = 0.0

                length_m = float(max(0.0, length_km) * 1000.0)

                out_rows.append((
                    agent_ids[a],
                    nodes[js_all[a]],
                    nodes[i],
                    int(day.strftime("%Y%m%d")),
                    (tbin.strftime("%Y-%m-%d %H:%M:%S")),
                    float(length_m),
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
    pep_df.sort_values(by=[DATE_COL, TIME_COL, START_COL, END_COL], inplace=True, ignore_index=True)

    if WRITE_RAW:
        raw_path = os.path.join(run_paths["run"], f"pep_raw_{tag}.csv")
        pep_df.to_csv(raw_path, index=False)
        info(f"Wrote raw PEP rows: {raw_path}")

    if WRITE_OD_AGGREGATE:
        g = pep_df.groupby([START_COL, END_COL, DATE_COL, TIME_COL], as_index=False)
        od = g.agg(trip_count=("length_m", "size"), m_length_m=("length_m", "mean"), mdn_length_m=("length_m", "median"))
        od.sort_values(by=[DATE_COL, TIME_COL, START_COL, END_COL], inplace=True, ignore_index=True)
        od_path = os.path.join(run_paths["run"], f"pep_od_{tag}.csv")
        od.to_csv(od_path, index=False)
        info(f"Wrote OD aggregate: {od_path}")

    # --- Population snapshots & aggregates (now initial AND final) ---
    def _counts_from_state(agent_state: np.ndarray, N: int) -> np.ndarray:
        return np.bincount(agent_state, minlength=N).astype(int)

    def _safe_share(counts_or_sum: np.ndarray | float, total: int) -> np.ndarray | float:
        denom = float(max(1, total)); return counts_or_sum / denom

    final_counts   = _counts_from_state(agent_state, N)

    centers_mask = is_center_mask
    df_tiles = pd.DataFrame({
        "geohash5": nodes,
        "initial_count": initial_counts,
        "initial_share": _safe_share(initial_counts, POOL_SIZE) * 100.0,
        "final_count": final_counts,
        "final_share": _safe_share(final_counts, POOL_SIZE) * 100.0,
        "delta_share": (_safe_share(final_counts, POOL_SIZE) - _safe_share(initial_counts, POOL_SIZE)) * 100.0,
        "is_center": centers_mask,
    }).sort_values(["is_center","final_count"], ascending=[False, False], ignore_index=True)

    path_tiles = os.path.join(run_paths["run"], f"pep_population_by_tile_{tag}.csv")
    df_tiles.to_csv(path_tiles, index=False)
    info(f"[WRITE] per-tile initial/final population: {path_tiles}")

    def _grp_sum(mask: np.ndarray, init_cnt: np.ndarray, fin_cnt: np.ndarray) -> Dict[str, float | int]:
        ic = int(init_cnt[mask].sum())
        fc = int(fin_cnt[mask].sum())
        return {
            "initial_count_total": ic,
            "initial_total_share": ic / float(max(1, POOL_SIZE)) * 100.0,
            "final_count_total": fc,
            "final_total_share": fc / float(max(1, POOL_SIZE)) * 100.0,
            "delta_total_share": (fc - ic) / float(max(1, POOL_SIZE)) * 100.0
        }

    agg_rows = []
    c_grp = _grp_sum(centers_mask, initial_counts, final_counts); c_grp["group"] = "center_set"
    nc_grp = _grp_sum(~centers_mask, initial_counts, final_counts); nc_grp["group"] = "non_center"
    tot_grp = {
        "group": "TOTAL",
        "initial_count_total": int(initial_counts.sum()),
        "initial_total_share": 100.0,
        "final_count_total": int(final_counts.sum()),
        "final_total_share": 100.0,
        "delta_total_share": 0.0
    }
    agg_rows += [c_grp, nc_grp, tot_grp]
    df_agg = pd.DataFrame(agg_rows, columns=[
        "group","initial_count_total","initial_total_share","final_count_total","final_total_share","delta_total_share"
    ])
    path_agg = os.path.join(run_paths["run"], f"pep_population_aggregates_{tag}.csv")
    df_agg.to_csv(path_agg, index=False)
    info(f"[WRITE] aggregates (initial & final): {path_agg}")

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
