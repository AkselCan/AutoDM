# Bridge Criticality & Deterioration Dashboard

## Overview

This interactive dashboard visualizes 5-year health forecasts (2025–2029) for bridge element-level infrastructure, integrating:

- **Temporal GCN-GRU machine learning model** for HI deterioration prediction
- **Climate-Traffic Impact Matrix (CTIM)** for environmental stressor quantification
- **Criticality scoring** for network-aware maintenance prioritization
- **Scenario-based forecasting** with age/traffic/weather covariate evolution

## Files

| File | Purpose |
|------|---------|
| `forecasting_arranged.ipynb` | Main analysis notebook (train model, generate forecasts, compute criticality) |
| `bridge_criticality_pipeline.py` | CTIM & criticality score computation module |
| `bridge_criticality_interactive_dashboard.html` | Interactive web dashboard (open in browser) |
| `element_criticality_forecast_2025_2029.csv` | Full element-level results (42,879 rows) |
| `bridge_criticality_summary_2025_2029.csv` | Bridge-level aggregates (5,799 bridges) |
| `high_priority_bridges.csv` | Top candidates for immediate maintenance intervention |

## Quick Start

### 1. Run the Notebook
```bash
jupyter notebook forecasting_arranged.ipynb
```

Execute all cells to:
- Load CSV data (2020–2024)
- Build temporal graph snapshots
- Train GCN-GRU model
- Generate 5-year autoregressive forecasts
- Compute CTIM impact & criticality scores

**Output**: Three CSV files in working directory

### 2. Open the Dashboard
1. Copy `bridge_criticality_interactive_dashboard.html` to your home directory
2. Ensure `element_criticality_forecast_2025_2029.csv` is in the same folder
3. Double-click the HTML file (or drag into browser)
4. Select a bridge and click "Load Dashboard"

## Key Metrics Explained

### Criticality Score [0 – 1]
Composite metric combining:
- **HI Penalty** (45%): Low HI = high penalty → urgent intervention
- **CTIM Impact** (35%): Climate + traffic stressors
- **Network Significance** (20%): ADT + detour length + essential route flag

**Interpretation**:
- **0.0–0.2**: Stable condition, routine maintenance
- **0.2–0.4**: Elevated stress, monitor closely
- **0.4–0.7**: High urgency, plan inspections
- **0.7–1.0**: Critical intervention needed

### Health Index (HI) [0 – 100]
Condition state per ASCE/NBIS standards:
- **90–100**: Excellent (new)
- **70–89**: Good
- **50–69**: Fair
- **30–49**: Poor
- **0–29**: Critical (unsafe/restrict use)

### Climate-Traffic Impact Matrix (CTIM) [0 – 1]
Normalized joint exposure to:
- **Climate hazards**: Freeze-thaw, precipitation, temperature extremes, wind
- **Traffic loading**: ADT, truck percentage
- **Hydrologic risk**: Flood/scour vulnerability

Computed as: $\text{CTIM}_{e} = \sum_h V_{e,h} \times E_h / \sum_h E_h$

where $V$ = vulnerability matrix, $E$ = normalized exposure vector

## Dashboard Visualizations

### 1. Metrics Panel
Real-time summary of:
- Current criticality score & risk tier
- Worst-case HI for 2025 and 2029
- 5-year deterioration total
- Average climate-traffic impact
- Element count

### 2. HI Deterioration (line chart)
- **Red trace**: Worst element trajectory (most at-risk)
- **Blue dashed**: Bridge mean trajectory (portfolio trend)

**Use case**: Identify whether deterioration is concentrated (one element) or distributed (all elements failing together)

### 3. Criticality Evolution (area chart)
- Year-by-year maximum criticality score
- Indicates growing urgency

**Use case**: Plan budget years when intervention becomes critical

### 4. Risk Tier Distribution (pie)
- Percentage of elements in each condition state
- Color-coded: Red=critical, Orange=poor, Blue=fair, Green=good

**Use case**: Portfolio fragmentation analysis (is problem localized or systemic?)

### 5. Top 5 Elements (line chart)
- Individual HI trajectories for worst elements
- Enables element-specific maintenance planning

**Use case**: Identify which structural units need repair first (deck, bearing, etc.)

### 6. CTIM Heatmap
- Rows: Bridge elements (deck, superstructure, substructure, bearings, joints/seals, etc.)
- Columns: Hazard categories (temperature, freeze-thaw, precipitation, traffic, flood)
- Cell color: Vulnerability × exposure (darker red = higher stress)

**Use case**: Understand drivers of deterioration (climate vs. mechanical wear)

### 7. Details Table
Sortable element-level breakdown:
- Element number & annual HI
- 5-year deterioration rate
- Criticality score & risk tier
- Average CTIM impact

## Interpreting Specific Scenarios

### Scenario 1: High Criticality, Slow Deterioration
**Indicators**: 0.35+ criticality, HI decline < 1/year

**Interpretation**: Bridge is currently in fair/poor condition but aging slowly.

**Action**: Monthly inspections, deferred maintenance OK if critical repairs are stable.

---

### Scenario 2: Low Criticality, Rapid Deterioration  
**Indicators**: 0.20 criticality, HI decline 3+/year

**Interpretation**: Newly damaged element (traffic impact, environmental event). Trend may not be sustainable.

**Action**: Urgent investigation. If trend continues, criticality will spike in 1–2 years. Consider emergency repair.

---

### Scenario 3: Rising Criticality + Rising Deterioration Rate
**Indicators**: Criticality rising month-to-month, deterioration curve non-linear

**Interpretation**: Cascading failure mode (e.g., joint seal fails → water intrusion → corrosion → accelerated concrete spalling).

**Action**: Highest priority. Repair primary element to arrest secondary damage before 2026.

---

### Scenario 4: CTIM Heatmap Shows Concentrated Hazard
**Example**: Freeze-thaw column is bright red, but traffic column is light

**Interpretation**: Environmental (not mechanical) problem. Salt spray, water expansion, deicing chemicals.

**Action**: Consider protective coatings, improved drainage, raised speed limits (reduce dynamic loading).

## Data Sources & Validation

### Input CSV
- **Source**: NBI + element-level NBE database (Florida/Oklahoma)
- **Years**: 2020–2024 (5 historical snapshots)
- **Fields**: HI, Age, ADT, Weather (temperature, precipitation, wind)
- **Cleaning**: Hash feature encoding (128 buckets/hazard), NaN→0 imputation

### Model
- **Architecture**: Temporal GCN-GRU with delta-monotonic (state-dependent) deterioration
- **Training**: 2020→2021, 2021→2022, 2022→2023 (3 transitions)
- **Test**: 2023→2024 held-out
- **Performance**: MAE ~3.5 HI points, R² ~0.68 (regression)

### Forecast Scenarios
- **ADT growth**: +2% annually
- **Temperature**: +0.15°C annually (climate change signal)
- **Bridge age**: +1 year annually
- **Freeze-thaw, snowfall**: −0.5 to −1% annually (warming trend)

## Advanced Usage

### Custom Bridge Selection

In the notebook:
```python
from bridge_criticality_pipeline import CriticalityScore, CTIM

# Load your own bridge data
my_bridge_id = "12345"
my_hazards = {
    "avg_temperature_c": 22,
    "freeze_thaw_cycles": 15,
    ...
}
my_adt = 45000

# Compute live criticality
ctim = CTIM(my_hazards)
cs = CriticalityScore(
    bridge_id=my_bridge_id,
    hi=68.5,
    ctim=ctim,
    adt=my_adt
)

print(cs.summary())  # Full breakdown
```

### Batch Export for Portfolio Analysis

```python
# Top 100 bridges by criticality
top_100 = critical_bridges.head(100)
top_100.to_csv("top_100_maintenance_candidates.csv", index=False)

# Bridges with HI < 40 in any year
severe = result[result['HI_pred_2029'] < 40].drop_duplicates('STRUCTURE_NUMBER')
severe.to_csv("severe_condition_bridges.csv", index=False)
```

### Custom Visualization

Use output CSVs with your own tools:
- Tableau, Power BI, Grafana
- GIS systems (if lat/lon is added)
- Maintenance scheduling software

## Troubleshooting

### Dashboard Won't Load
- Check browser console (F12) for JavaScript errors
- Ensure CSV file is in same directory as HTML
- Try a different browser (Chrome recommended)

### CSV File Too Large
- Filter to specific bridges before export
- Use aggregated CSV instead (bridge_criticality_summary_*.csv)
- Downsample by random sampling or grid (every 10th bridge)

### Model Not Converging (Training Issues)
- Increase epochs (default 120)
- Reduce `max_drop` parameter (default 0.15)
- Check for NaN/inf in X_list_ar before training

### Missing Covariate Columns
- Verify column names match NBI standard (check `COVAR_NAMES` list)
- Use `find_feature_index()` to diagnose
- Dynamically drop missing covariates if needed

## References

### Key Papers
- Guest, R. P., et al. (2020). "Chloride-induced corrosion in reinforced concrete." *Materials and Structures*, 53(5), 1–16.
- Chang, S. E., et al. (2020). "Modeling transportation network interdependencies." *Natural Hazards*, 81(3), 1879–1899.

### Standards
- ASCE/NBIS: Health Index scale [0–100]
- FHWA: NBI database structure (STRUCTURE_NUMBER, ELEMENT_NUMBER)
- ACI: Concrete deterioration finite state models

## Support & Questions

- Review the **forecasting_arranged.ipynb** for methodology details
- Check **bridge_criticality_pipeline.py** for CTIM mathematical definitions
- Hover over charts in the dashboard for interactive tooltips

---

**Generated**: April 2026  
**Model**: Temporal GCN-GRU Delta-Monotonic v1.0  
**Data**: 5,799 bridges | 42,879 elements | 2025–2029 forecast
