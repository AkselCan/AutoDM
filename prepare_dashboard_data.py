"""
prepare_dashboard_data.py
=========================
Preprocesses bridge NBI + element forecast CSVs into a compact dashboard_data.json.
Run after the forecasting notebook and before opening the dashboard:
    python prepare_dashboard_data.py
    python prepare_dashboard_data.py --input data/another_file.csv
"""

import argparse
import json
import math
import re
from datetime import date
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
RESULTS = BASE / "results"

parser = argparse.ArgumentParser(description="Build dashboard_data.json from the forecast results.")
parser.add_argument(
    "--input", type=Path, default=BASE / "data" / "my_state_merged.csv",
    help="merged NBE + NBI CSV (default: data/my_state_merged.csv)",
)
args = parser.parse_args()

ELEMENT_NAMES = {
    10: "Wearing Surface",
    12: "RC Deck", 13: "PC Deck", 28: "Timber Deck",
    30: "Steel Deck (Open Grid)", 31: "Steel Deck (Corrugated)",
    32: "Steel Deck (Concrete Filled)", 38: "RC Slab", 39: "PC Slab",
    52: "PC Top Flange", 53: "RC Top Flange", 54: "Steel Top Flange",
    101: "Steel Truss (Comp.)", 102: "RC Arch", 103: "PC Arch",
    104: "Steel Arch", 105: "Other Arch", 106: "Steel Truss",
    107: "Elastomeric Bearing", 109: "Disk Bearing",
    110: "Other Bearing", 111: "Enclosed Bearing",
    112: "Fixed Bearing", 113: "Movable Bearing", 115: "Pot Bearing",
    120: "Pourable Joint Seal", 121: "Comp. Joint Seal",
    122: "Modular Joint Seal", 123: "Open Expansion Joint",
    124: "Assembly Joint (No Seal)", 125: "Strip Seal Expansion Joint",
    131: "Other Joint",
    152: "Steel Bridge Rail", 153: "PC Bridge Rail",
    154: "RC Bridge Rail", 155: "Timber Bridge Rail", 156: "Other Bridge Rail",
    160: "RC Abutment", 161: "Masonry Abutment", 162: "Other Abutment",
    170: "RC Pier Cap", 171: "PC Pier Cap", 172: "Steel Pier Cap",
    173: "Timber Pier Cap", 174: "Other Pier Cap",
    175: "RC Pier Wall", 176: "Steel Column", 177: "RC Column",
    178: "PC Column", 179: "Timber Column",
    181: "RC Pile", 182: "PC Pile", 183: "Steel Pile", 184: "Timber Pile",
    186: "RC Pile Cap/Footing", 187: "Other Pier Wall",
    202: "Steel Girder/Beam", 203: "PC Girder/Beam", 204: "RC Girder/Beam",
    205: "PC Floor Beam", 206: "Steel Floor Beam", 207: "RC Floor Beam",
    208: "PC Closed Web/Box Girder", 209: "Steel Closed Web/Box Girder",
    210: "RC Closed Web/Box Girder",
    211: "PC Stringer", 212: "Steel Stringer", 213: "RC Stringer",
    214: "Other Girder/Beam",
    215: "Steel Gusset Plate", 216: "Steel Main Cable",
    217: "Steel Secondary Cable",
    220: "Concrete Protective Coating", 221: "Steel Protective Coating",
    225: "PC Pile", 226: "RC Culvert (PC)", 231: "Steel Culvert",
    234: "PC Pier Cap", 241: "RC Culvert",
    301: "Strip Seal Joint", 302: "Compression Joint Seal",
    310: "Wearing Surfaces", 311: "Concrete Prot. Coating",
    313: "Steel Prot. Coating", 315: "Pourable Joint Seal",
    320: "Timber Pile Cap", 321: "Timber Pile",
    322: "Timber Abutment", 323: "Timber Pier Cap",
    324: "Timber Column", 325: "Timber Stringer",
    326: "Timber Deck", 327: "Timber Bridge Rail",
    330: "Pin/Hanger Assembly", 331: "RC Bridge Rail",
    510: "Wearing Surfaces",
    515: "Steel Prot. Coating",
}

KEY_NBI = [
    'STRUCTURE_NUMBER', 'YEAR',
    'NBI_16 - Latitude (decimal)', 'NBI_17 - Longitude (decimal)',
    'NBI_22 - Owner Agency', 'NBI_27 - Year Built',
    'NBI_29 - Average Daily Traffic', 'NBI_43A - Main Span Material',
    'NBI_43B - Main Span Design', 'NBI_6A - Features Intersected',
    'NBI_7 - Facility Carried By Structure', 'NBI_9 - Location',
    'NBI_Bridge Age (yr)', 'NBI_49 - Structure Length (ft.)',
    'NBI_52 - Deck Width - Out to Out (ft.)',
    'NBI_CAT29 - Deck Area (sq. ft.)', 'NBI_3 - County Name',
    'STATE_NAME', 'NBI_CAT10 - Bridge Condition',
    'NBI_Average Temperature', 'NBI_Maximum Temperature',
    'NBI_Minimum Temperature', 'NBI_Number of Freeze-Thaw Cycles',
    'NBI_Average Relative Humidity',
    'NBI_109 - Average Daily Truck Traffic (Percent ADT)',
    'NBI_Total Precipitation',
    'NBI_Number of Days with Measurable Precipitation',
    'NBI_Mean Wind Speed', 'NBI_Number of Snowfall Days',
    'NBI_58 - Deck Condition Rating',
    'NBI_59 - Superstructure Condition Rating',
    'NBI_60 - Substructure Condition Rating',
    'NBI_21 - Maintenance Responsibility',
    'NBI_114 - Future Average Daily Traffic',
    'NBI_113 - Scour Critical Bridge Value',
    'NBI_34 - Skew Angle (degrees)',
    'NBI_45 - Number of Spans in Main Unit',
    'NBI_106 - Year Reconstructed',
]

def safe_float(v, default=None):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default

def safe_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default

def safe_str(v, default=""):
    if pd.isna(v):
        return default
    return str(v).strip()

def normalize_bridge_id(v):
    """Match the notebook: numeric IDs lose padding and formatting, others stay as text."""
    if pd.isna(v):
        return v
    v = str(v).strip()
    try:
        return str(int(float(v)))
    except (ValueError, OverflowError):
        return v

def within_main_extent(values, margin=2.0):
    """Drop stray coordinates more than `margin` degrees beyond the central 98% of bridges."""
    low, high = values.quantile([0.01, 0.99])
    return values.between(low - margin, high + margin)


print(f"Loading NBI data from {args.input} …")
available = pd.read_csv(args.input, nrows=0).columns.tolist()
nbi_df = pd.read_csv(
    args.input,
    usecols=[c for c in KEY_NBI if c in available], dtype=str, low_memory=False
)
nbi_df['STRUCTURE_NUMBER'] = nbi_df['STRUCTURE_NUMBER'].apply(normalize_bridge_id)
nbi_df['YEAR'] = nbi_df['YEAR'].str.strip()

# Keep most-recent year per bridge
nbi_df = (nbi_df
          .sort_values('YEAR', ascending=False)
          .drop_duplicates(subset='STRUCTURE_NUMBER', keep='first'))

print(f"  Unique bridges: {len(nbi_df)}")

print("Loading bridge criticality summary …")
sum_df = pd.read_csv(RESULTS / "bridge_criticality_summary_2025_2029.csv", dtype=str)
sum_df['STRUCTURE_NUMBER'] = sum_df['STRUCTURE_NUMBER'].apply(normalize_bridge_id)

print("Loading element-level HI forecasts …")
elem_df = pd.read_csv(RESULTS / "element_criticality_forecast_2025_2029.csv", dtype=str)
elem_df['STRUCTURE_NUMBER'] = elem_df['STRUCTURE_NUMBER'].apply(normalize_bridge_id)
elem_df['ELEMENT_NUMBER'] = elem_df['ELEMENT_NUMBER'].str.strip()

# Merge
merged = nbi_df.merge(sum_df, on='STRUCTURE_NUMBER', how='inner')
print(f"  Bridges with criticality scores: {len(merged)}")

# Filter out bridges with invalid lat/lon
lat_col = 'NBI_16 - Latitude (decimal)'
lon_col = 'NBI_17 - Longitude (decimal)'
merged = merged[merged[lat_col].notna() & merged[lon_col].notna()]
merged['_lat'] = pd.to_numeric(merged[lat_col], errors='coerce')
merged['_lon'] = pd.to_numeric(merged[lon_col], errors='coerce')
merged = merged[within_main_extent(merged['_lat']) & within_main_extent(merged['_lon'])]
print(f"  Bridges with valid coordinates: {len(merged)}")

# Build per-bridge element lookup
print("Building element lookup …")
elem_by_bridge = {}
for struct_id, grp in elem_df.groupby('STRUCTURE_NUMBER'):
    elems = []
    for _, row in grp.iterrows():
        try:
            en = int(row['ELEMENT_NUMBER'])
        except Exception:
            continue
        elems.append({
            "en": en,
            "name": ELEMENT_NAMES.get(en, f"Element {en}"),
            "hi": [
                safe_float(row.get('HI_pred_2025')),
                safe_float(row.get('HI_pred_2026')),
                safe_float(row.get('HI_pred_2027')),
                safe_float(row.get('HI_pred_2028')),
                safe_float(row.get('HI_pred_2029')),
            ],
            "cs": safe_float(row.get('criticality_score')),
        })
    # Sort by worst projected HI (lowest 2029) to surface most degraded first
    elems.sort(key=lambda e: (e['hi'][4] if e['hi'][4] is not None else 999))
    elem_by_bridge[struct_id] = elems

print("Serialising bridges …")
bridges = []
for _, row in merged.iterrows():
    sid = row['STRUCTURE_NUMBER']
    elems = elem_by_bridge.get(sid, [])

    # Compute HI trajectory for the whole bridge (avg of all elements)
    hi_series = [None, None, None, None, None]
    if elems:
        for yr in range(5):
            vals = [e['hi'][yr] for e in elems if e['hi'][yr] is not None]
            hi_series[yr] = round(float(np.mean(vals)), 2) if vals else None

    bridge = {
        "id": sid,
        "lat": round(row['_lat'], 6),
        "lon": round(row['_lon'], 6),
        # Identity
        "facility": safe_str(row.get('NBI_7 - Facility Carried By Structure')),
        "feature": safe_str(row.get('NBI_6A - Features Intersected')),
        "location": safe_str(row.get('NBI_9 - Location')),
        "county": safe_str(row.get('NBI_3 - County Name')),
        "state": re.sub(r'^\d+\s*-\s*', '', safe_str(row.get('STATE_NAME'))),
        # Physical
        "year_built": safe_int(row.get('NBI_27 - Year Built')),
        "year_recon": safe_int(row.get('NBI_106 - Year Reconstructed')),
        "age": safe_int(row.get('NBI_Bridge Age (yr)')),
        "length_ft": safe_float(row.get('NBI_49 - Structure Length (ft.)')),
        "width_ft": safe_float(row.get('NBI_52 - Deck Width - Out to Out (ft.)')),
        "deck_area_sf": safe_float(row.get('NBI_CAT29 - Deck Area (sq. ft.)')),
        "spans": safe_int(row.get('NBI_45 - Number of Spans in Main Unit')),
        "skew_deg": safe_float(row.get('NBI_34 - Skew Angle (degrees)')),
        "material": safe_str(row.get('NBI_43A - Main Span Material')),
        "design": safe_str(row.get('NBI_43B - Main Span Design')),
        # Admin
        "owner": safe_str(row.get('NBI_22 - Owner Agency')),
        "maintenance": safe_str(row.get('NBI_21 - Maintenance Responsibility')),
        "condition": safe_str(row.get('NBI_CAT10 - Bridge Condition')),
        # NBI ratings
        "deck_rating": safe_int(row.get('NBI_58 - Deck Condition Rating')),
        "super_rating": safe_int(row.get('NBI_59 - Superstructure Condition Rating')),
        "sub_rating": safe_int(row.get('NBI_60 - Substructure Condition Rating')),
        "scour": safe_str(row.get('NBI_113 - Scour Critical Bridge Value')),
        # Traffic
        "adt": safe_int(row.get('NBI_29 - Average Daily Traffic')),
        "future_adt": safe_int(row.get('NBI_114 - Future Average Daily Traffic')),
        "truck_pct": safe_float(row.get('NBI_109 - Average Daily Truck Traffic (Percent ADT)')),
        # Climate / weather
        "avg_temp": safe_float(row.get('NBI_Average Temperature')),
        "max_temp": safe_float(row.get('NBI_Maximum Temperature')),
        "min_temp": safe_float(row.get('NBI_Minimum Temperature')),
        "freeze_thaw": safe_float(row.get('NBI_Number of Freeze-Thaw Cycles')),
        "humidity": safe_float(row.get('NBI_Average Relative Humidity')),
        "precip_mm": safe_float(row.get('NBI_Total Precipitation')),
        "precip_days": safe_float(row.get('NBI_Number of Days with Measurable Precipitation')),
        "wind_speed": safe_float(row.get('NBI_Mean Wind Speed')),
        "snowfall_days": safe_float(row.get('NBI_Number of Snowfall Days')),
        # Criticality
        "crit_score": safe_float(row.get('criticality_score_max')),
        "risk_tier": safe_str(row.get('risk_tier')),
        "ctim": safe_float(row.get('ctim_impact_mean')),
        "hi_penalty": safe_float(row.get('hi_penalty_mean')),
        # HI trajectory (bridge-level mean)
        "hi": hi_series,
        # Elements (top degrading first, cap at 20 to keep JSON lean)
        "elements": elems[:20],
    }
    bridges.append(bridge)

# Summary stats for the scenario panel
risk_counts = {"critical": 0, "poor": 0, "fair": 0, "good": 0}
for b in bridges:
    t = b.get("risk_tier", "fair")
    if t in risk_counts:
        risk_counts[t] += 1

output = {
    "generated": date.today().isoformat(),
    "total_bridges": len(bridges),
    "risk_counts": risk_counts,
    "years": [2025, 2026, 2027, 2028, 2029],
    "bridges": bridges,
}

out_path = BASE / "dashboard_data.json"
with open(out_path, "w") as f:
    json.dump(output, f, separators=(',', ':'))

size_mb = out_path.stat().st_size / 1_048_576
print(f"\nDone. Written {out_path}  ({len(bridges)} bridges, {size_mb:.1f} MB)")
