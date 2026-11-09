# AutoDM

AutoDM forecasts element-level bridge health for 2025–2029 with a temporal graph neural network, scores maintenance criticality, and shows the results on an interactive map.

**AutoDM does not include a dataset.** You bring your own National Bridge Inventory (NBI) and National Bridge Elements (NBE) data. All `*.csv` files are ignored by git, so your data stays on your machine.

```
Your NBE + NBI CSV files
   |   Step 2: merge into one file
   v
data/my_state_merged.csv
   |   Step 3: notebooks/forecasting_arranged.ipynb
   v
results/*.csv and models/*.pt
   |   Step 4: prepare_dashboard_data.py
   v
dashboard_data.json
   |   Step 5: serve_dashboard.py
   v
Interactive dashboard in your browser
```

## Contents

- [Quick look without data](#quick-look-without-data)
- [Step 1: Install](#step-1-install)
- [Step 2: Prepare your data](#step-2-prepare-your-data)
- [Step 3: Run the forecast](#step-3-run-the-forecast)
- [Step 4: Build the dashboard data](#step-4-build-the-dashboard-data)
- [Step 5: Open the dashboard](#step-5-open-the-dashboard)
- [Using the dashboard](#using-the-dashboard)
- [Optional components](#optional-components)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Other documentation](#other-documentation)
- [Resources](#resources)

## Quick look without data

The repository includes a sample `dashboard_data.json` for Oklahoma, so you can see the dashboard before preparing any data. You only need Python:

```bash
git clone https://github.com/AkselCan/AutoDM.git
cd AutoDM
python serve_dashboard.py
```

On macOS or Linux, use `python3` if `python` is not found.

The map opens over Florida, so zoom out and pan to Oklahoma to see the bridges. Use the **Bridge Avg HI** color metric for this sample. Its criticality scores are on a 0–100 scale, but the dashboard expects 0–1, so the **Criticality Score** metric shows every bridge as critical.

## Step 1: Install

You need Python 3.11 or 3.12. If you already cloned the repository for the quick look, skip the `git clone` and `cd` lines.

```bash
git clone https://github.com/AkselCan/AutoDM.git
cd AutoDM
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install torch torch_geometric numpy "pandas<3" scipy scikit-learn matplotlib jupyter
```

A GPU is optional and is used automatically if PyTorch can see one. For CUDA support, install PyTorch first using the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

## Step 2: Prepare your data

### What you need

- An NBE export and an NBI export for the same bridges.
- **Five consecutive years of data**, 2020–2024 by default.

The NBI column names used here follow FHWA [LTBP InfoBridge](https://infobridge.fhwa.dot.gov/) exports.

### Merge NBE and NBI

The model reads one CSV with one row per NBE element record and year, with NBI columns prefixed `NBI_`. If an element appears more than once for a bridge and year, the last row is used.

The script below builds that file. It needs these columns in your exports:

- **NBE:** `YEAR`, `STRUCTURE_NUMBER`, `ELEMENT_CATEGORY`, `PARENT_ELEMENT_NAME`, `ELEMENT_NAME`
- **NBI:** `Year`, `8 - Structure Number`

Save it as `merge.py` in the repository root, set `NBE_CSV` and `NBI_CSV`, and run `python merge.py` with your virtual environment active. Steps 3 and 4 assume you leave `OUT_CSV` unchanged.

```python
import pandas as pd

NBE_CSV = "path/to/your_nbe.csv"
NBI_CSV = "path/to/your_nbi.csv"
OUT_CSV = "data/my_state_merged.csv"

nbe = pd.read_csv(NBE_CSV, dtype=str)
nbi = pd.read_csv(NBI_CSV, dtype=str)

# One-hot encode the element descriptors; the model uses them as features.
nbe = pd.concat([
    nbe,
    pd.get_dummies(nbe["ELEMENT_CATEGORY"], prefix="CAT"),
    pd.get_dummies(nbe["PARENT_ELEMENT_NAME"], prefix="PARENT"),
    pd.get_dummies(nbe["ELEMENT_NAME"], prefix="ELEM"),
], axis=1)

# Join on bridge ID + year. IDs are normalized because exports may pad or reformat them.
def norm_id(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()

nbe["_id"], nbe["_year"] = nbe["STRUCTURE_NUMBER"].map(norm_id), pd.to_numeric(nbe["YEAR"])
nbi["_id"], nbi["_year"] = nbi["8 - Structure Number"].map(norm_id), pd.to_numeric(nbi["Year"])

nbi = nbi.drop_duplicates(["_id", "_year"], keep="last")
nbi = nbi.rename(columns={c: f"NBI_{c}" for c in nbi.columns if c not in ("_id", "_year")})

merged = nbe.merge(nbi, on=["_id", "_year"], how="left", validate="m:1")
merged.drop(columns=["_id", "_year"]).to_csv(OUT_CSV, index=False)
print(f"{len(merged):,} rows, {merged['STRUCTURE_NUMBER'].nunique():,} bridges -> {OUT_CSV}")
```

> Keep bridge IDs as text. Opening a CSV in Excel and saving it can turn `STRUCTURE_NUMBER` into values like `3.271E+14`, which permanently loses digits.

### Column reference

| Column | Needed | Notes |
|---|---|---|
| `YEAR` | Required | Snapshot year |
| `STRUCTURE_NUMBER` | Required | Bridge ID |
| `ELEMENT_NUMBER` | Required | Must be an integer |
| `HI` | Required | Element health index, 0–100. AutoDM does not calculate this. If your NBE export lacks it, `CS_PERCENT_1 + 0.67 * CS_PERCENT_2 + 0.33 * CS_PERCENT_3` closely approximates the values in the project's data |
| `NBI_16 - Latitude (decimal)`, `NBI_17 - Longitude (decimal)` | Required for the dashboard | Bridges without valid coordinates are dropped |
| `PARENT_ELEMENT_NUMBER` | Recommended | Adds parent–child element links to the graph |
| Climate and traffic columns, listed below | Recommended | Model features and inputs to the future scenarios and climate-traffic scores, though [not all are in use yet](#known-limitations). Missing ones are reported and skipped |

Climate and traffic columns:

```
NBI_Bridge Age (yr)
NBI_29 - Average Daily Traffic
NBI_Average Temperature
NBI_Maximum Temperature
NBI_Minimum Temperature
NBI_Number of Freeze-Thaw Cycles
NBI_Number of Days with Measurable Precipitation
NBI_Number of Days with Temperature Below 0?C
NBI_Number of Snowfall Days
NBI_Total Precipitation
```

The `?` in `Below 0?C` is a literal character in the column name.

Other numeric columns become model features automatically, and ID columns are excluded. Text columns are hashed (see [Known limitations](#known-limitations)). Missing numeric values are treated as 0, and rows with no `HI` are left out of training.

### Memory

Large states need a lot of memory. A Florida-sized dataset (about 43,000 elements per year) needs roughly 20 GB of RAM. That figure is estimated from array sizes rather than measured. If you run out of memory, lower the `n_features=128` value in the `hash_categorical_features(df, n_features=128)` call inside `make_feature_frame`, in section 1 of the notebook.

## Step 3: Run the forecast

1. Start Jupyter **from inside `notebooks/`**, because every path in the notebook is relative to that folder:

   ```bash
   cd notebooks
   jupyter notebook forecasting_arranged.ipynb
   ```

2. In section **0) Setup**, point `CSV_PATH` at your file:

   ```python
   CSV_PATH = "../data/my_state_merged.csv"
   ```

   If your data does not cover 2020–2024, also change `YEARS` here. Using a different *number* of years also requires changing the train and test indices in sections 5 and 6.

   The forecast years 2025–2029 are written into the cell that starts with `scenario = {`, the later cells that use `HI_pred_2025` through `HI_pred_2029`, `prepare_dashboard_data.py` (lines 171–175 and 263), and `YEARS` in the dashboard HTML. Changing them means updating all of those.

3. Run the cells in this order. **Do not use Run All**, because one cell depends on a cell further down the notebook.

   | Order | Cells | Result |
   |---|---|---|
   | 1 | Sections 0–7, from the top through the cell that starts with `scenario = {` | Trains the model and forecasts 2025–2029 |
   | 2 | The cell that starts with `# Build bridge_hazards_dict and bridge_meta_dict`, located after section 10 | Builds the climate and traffic inputs |
   | 3 | Back up to the cell that starts with `from bridge_criticality_pipeline import`, then the two cells after it | Scores criticality and saves the results |
   | 4 | *Optional:* the cell that starts with `# Advanced filtering utility` | Saves `high_priority_bridges.csv` |
   | 5 | *Optional:* section 8 | Plots trajectories and the 2029 distribution |

   Skip the last cell in the notebook, which starts with `import pandas as pd` and loops over a `("TX", "NEW_NBE_and_NBI_OK.csv")` entry. It is a statistics check for one specific local file. Sections 9 and 10 are older dashboard notes, so ignore their instructions. Apart from the optional filtering cell, their code cells only print information and can be skipped.

4. Check that these files were written:

   | File | Contents |
   |---|---|
   | `results/element_hi_forecast_2025_2029_delta_monotonic_scenario_FLO.csv` | Element HI forecast |
   | `results/element_criticality_forecast_2025_2029.csv` | Element forecast with criticality scores |
   | `results/bridge_criticality_summary_2025_2029.csv` | One row per bridge |
   | `results/high_priority_bridges.csv` | Top 20 bridges by criticality, if you ran the optional cell |
   | `models/temporal_gnn_bridge_hi_delta_monotonic.pt` | Trained model |

The notebook always trains a new model. The checkpoints in `models/` are not reloaded and only fit the dataset they were trained on.

## Step 4: Build the dashboard data

`prepare_dashboard_data.py` currently has one dataset's paths and map bounds written in. Make these four edits, then run it:

| Line | Current code | Change to |
|---|---|---|
| 116 | `ok_nbi_path = NOTEBOOKS / "NEW_NBE_and_NBI_OK.csv"` | `ok_nbi_path = DATA / "my_state_merged.csv"` |
| 136 | `pd.read_csv(NOTEBOOKS / "bridge_criticality_summary_2025_2029.csv", dtype=str)` | Replace `NOTEBOOKS` with `RESULTS` |
| 140 | `pd.read_csv(NOTEBOOKS / "element_criticality_forecast_2025_2029.csv", dtype=str)` | Replace `NOTEBOOKS` with `RESULTS` |
| 154 | `.between(33.6, 37.1)` and `.between(-103.1, -94.4)` | Your state's latitude and longitude ranges. Bridges outside them are dropped |

From the repository root:

```bash
python prepare_dashboard_data.py
```

This replaces `dashboard_data.json` with your bridges. To center the map on your state, also edit `setView([27.5, -82.5], 7)` in `bridge_criticality_interactive_dashboard.html`, on line 819.

> **This script needs numeric bridge IDs.** It converts each one with `int(float(x))` and stops with a `ValueError` on an ID such as `99AA99`.

> **Before committing:** following these steps changes tracked files, which then show up in `git status`: `dashboard_data.json`, `models/temporal_gnn_bridge_hi_delta_monotonic.pt`, `notebooks/forecasting_arranged.ipynb`, and `prepare_dashboard_data.py`, plus the dashboard HTML if you edited `setView`. Leave them out of your commits unless you mean to publish them.

## Step 5: Open the dashboard

```bash
python serve_dashboard.py
```

This serves the repository on port 8000 and opens this page in your browser:

`http://localhost:8000/bridge_criticality_interactive_dashboard.html`

Press `Ctrl+C` in the terminal to stop it.

- Opening the HTML file directly does not work. It has to be served.
- You need an internet connection, because Leaflet, Plotly, and the map tiles load from the web.

## Using the dashboard

The dashboard has three tabs. It rates each bridge from its health index: **Critical** below 50, **Poor** 50–70, **Fair** 70–85, and **Good** 85 and above.

**Map View**
- Choose what colors the markers: Bridge Avg HI, Worst Element HI, or Criticality Score.
- Use the checkboxes to show or hide tiers, and the slider to pick a forecast year from 2025 to 2029.
- Click a marker to see the bridge's condition ratings, criticality, worst elements, climate and traffic data, and ownership. You can also add it to a comparison from there.

**Compare**
- Compare up to three bridges side by side. Adding a fourth replaces the oldest.
- Search by bridge ID, facility, or county, then choose a suggestion. The Add button and Enter only accept an exact bridge ID.
- Each bridge shows its 2025 and 2029 HI and the change between them, along with a chart of its bridge average and up to seven worst elements.

**Scenarios**
- Choose a preset (Baseline, Climate Stress, Heavy Traffic, Extreme Weather, Optimistic) or set your own values with the sliders.
- **Apply to Map** carries the scenario over to the map, and **Reset to Baseline** clears it. Scenarios change Bridge Avg HI and Worst Element HI, but not Criticality Score or the per-element values in the bridge panel.
- Scenarios are a quick estimate that scales the forecast's year-over-year HI drops in the browser. They do not re-run the model.

The dashboard has no export or download feature, and comparisons and scenarios are lost when the page reloads.

## Optional components

You don't need any of these to run the steps above.

### Forecast with uncertainty

`notebooks/forecasting_arranged_mc_dropout_updated.ipynb` extends the main notebook with Monte Carlo dropout and baseline comparisons. It uses `scripts/bridge_criticality_pipeline_updated.py`, which scores criticality from 0 to 100. The dashboard's criticality views expect 0–1, so use the main notebook if you want to feed the dashboard. This notebook overwrites the main notebook's result CSVs and model checkpoint, so re-run the main notebook before Step 4 if you have run this one.

`notebooks/forecasting.ipynb` is an early baseline kept for reference. It does not run from top to bottom.

### Static graph model

This separate track trains a bridge-level graph model from a Neo4j database. It needs a running Neo4j instance at `bolt://localhost:7687` and these extra packages:

```bash
pip install neo4j rdflib rdflib-neo4j
```

> **Warning:** the code under **Newer version w/ neo4j compatability** in `notebooks/ontologyBuilder.ipynb` runs `MATCH (n) DETACH DELETE n`, which deletes everything in the connected database. Use a dedicated Neo4j instance.

1. Set your Neo4j connection details in `notebooks/ontologyBuilder.ipynb`, under both **Newer version w/ neo4j compatability** and **Data population**, and at the top of `scripts/no_torch.py`.
2. Under **Data population**, set `csv_file_path = "../data/my_state_merged.csv"`, with Jupyter started from `notebooks/`.
3. Run the cells under those two headings, in order.
4. From the repository root, run:

   ```bash
   python scripts/no_torch.py
   python scripts/bridge_extract_toTorch.py
   python scripts/train_bridge_gnn.py
   ```

### Streamlit dashboard

`scripts/dashboard.py` is a separate app for exploring the Neo4j graph, with an AI assistant powered by Gemini. It does not show the forecasts. Replace the Neo4j and Gemini settings at the top of the file with your own, then run:

```bash
pip install streamlit pydeck plotly neo4j google-generativeai
streamlit run scripts/dashboard.py
```

## Known limitations

These apply to `notebooks/forecasting_arranged.ipynb` and affect how you should read its results.

- **Criticality does not use the forecast yet.** Scoring looks for an HI column the forecast does not produce and falls back to a fixed HI of 70. Scores reflect climate and traffic only, and every element's risk tier is `fair`.
- **Forecast year labels run ahead of the data.** The columns labeled 2025–2029 effectively cover about 2024–2026.
- **Few inputs change across forecast years.** Only average daily traffic, snowfall days, and total precipitation change. Bridge age, temperature, and freeze-thaw cycles stay at their last observed values.
- **Some hazards are missing from the climate-traffic scores.** Below-zero days, wind speed, truck percentage, and flood risk are always counted as zero.
- **Text columns add no information.** The hashing step gives every text value the same result, so these columns only use memory.
- **Some summary columns are empty.** `worst_hi_2025` and `worst_hi_2029` in the bridge summary, and the worst-HI and deterioration columns in `high_priority_bridges.csv`, are blank.
- **Elements with no recent HI are forecast as 0.** They lower their bridge's average and look critical on the dashboard.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` in the notebook | Start Jupyter from `notebooks/` and check `CSV_PATH` |
| `ModuleNotFoundError: bridge_criticality_pipeline` | Start Jupyter from `notebooks/`, where the module is located |
| `NameError: bridge_hazards_dict` | Run the cells in the order given in [Step 3](#step-3-run-the-forecast) |
| `IndexError` in the section 8 plots | The first plot needs at least 501 element rows. Lower `.iloc[500]` or skip the plot |
| `FileNotFoundError` from `prepare_dashboard_data.py` | Make the path edits in [Step 4](#step-4-build-the-dashboard-data) |
| `ValueError` from `prepare_dashboard_data.py` | Your data has non-numeric bridge IDs, which this script does not support |
| No bridges in `dashboard_data.json` | Check the latitude and longitude ranges on line 154 |
| "Failed to load dashboard data" | Start the page with `python serve_dashboard.py` and confirm `dashboard_data.json` exists |
| Blank page | Check your internet connection. The map and chart libraries load from the web |
| Port 8000 is already in use (on Windows, `WinError 10048`) | Run `python -m http.server 8001`, then open `http://localhost:8001/bridge_criticality_interactive_dashboard.html` |
| Out of memory during training | See [Memory](#memory) |

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Forecasting and ontology notebooks, plus `bridge_criticality_pipeline.py` |
| `scripts/` | Static graph pipeline, updated criticality module, Streamlit app |
| `data/` | Put your merged CSV here. Also holds exported graph data and metadata |
| `results/` | Forecast CSVs and training outputs |
| `models/` | Trained model weights |
| `ontology/` | OWL ontology files |
| `prepare_dashboard_data.py` | Builds `dashboard_data.json` |
| `serve_dashboard.py` | Serves the dashboard locally |
| `bridge_criticality_interactive_dashboard.html` | The dashboard page |

## Other documentation

These guides were written for an earlier version of the dashboard. Where they differ from this README, follow this README.

- [CRITICALITY_DASHBOARD_README.md](CRITICALITY_DASHBOARD_README.md): background on the health index, the Climate-Traffic Impact Matrix (CTIM), and the criticality score. Its chart descriptions, file locations, and model accuracy figures are out of date.
- [DASHBOARD_SETUP.md](DASHBOARD_SETUP.md): older setup and troubleshooting notes. The CSV upload option it describes no longer exists, and the dashboard now loads `dashboard_data.json`.

## Resources

- [Literature review progress](https://docs.google.com/spreadsheets/d/1FBF4gexEheqrDxaIwI9gxIBC1Uaw7EUmpFTdzaxkVrw/edit?usp=sharing): reviewed papers, key findings, and connections to this project.
- [FHWA Bridge Management Guide (PDF)](https://www.fhwa.dot.gov/bridge/mtguide.pdf): U.S. Federal Highway Administration manual on bridge management practices.
- [OU SharePoint folder](https://sooners-my.sharepoint.com/:f:/g/personal/asare_ou_edu/EngYxZUOt91DrwwzmfVf-MkB60QK_hNiGWCroPMJtDd-IA?email=asare%40ou.edu&e=L4nR1v): the research team's working files. You don't need these to run AutoDM.
