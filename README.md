# AutoDM

AutoDM forecasts element-level bridge health for 2025–2029 with a temporal graph neural network, scores maintenance criticality, and shows the results on an interactive map.

**AutoDM does not include a dataset.** You bring your own National Bridge Inventory (NBI) and National Bridge Elements (NBE) data. All `*.csv` files are ignored by git, so your data stays on your machine.

<!-- Demo GIF: uncomment after adding the file.
![AutoDM map view](docs/images/map-view.gif)
-->

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
- [How the scores work](#how-the-scores-work)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Resources](#resources)

## Quick look without data

The repository includes a sample `dashboard_data.json` for Oklahoma, so you can see the dashboard before preparing any data. You only need Python:

```bash
git clone https://github.com/AkselCan/AutoDM.git
cd AutoDM
python serve_dashboard.py
```

On macOS or Linux, use `python3` if `python` is not found.

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

2. If your merged file is not `data/my_state_merged.csv`, update `CSV_PATH` in section **0) Setup**.

   If your data does not cover 2020–2024, change `YEARS` there too. Using a different *number* of years also requires changing the train and test indices in sections 5 and 6.

   The forecast years 2025–2029 are written into the cell that starts with `scenario = {`, the later cells that use `HI_pred_2025` through `HI_pred_2029`, the matching names and `years` list in `prepare_dashboard_data.py`, and `YEARS` in the dashboard HTML. Changing them means updating all of those.

3. Run all cells from top to bottom. The notebook trains the model, forecasts 2025–2029, scores criticality, and saves the results.

4. Check that these files were written:

   | File | Contents |
   |---|---|
   | `results/element_hi_forecast_2025_2029_delta_monotonic_scenario_FLO.csv` | Element HI forecast |
   | `results/element_criticality_forecast_2025_2029.csv` | Element forecast with criticality scores |
   | `results/bridge_criticality_summary_2025_2029.csv` | One row per bridge |
   | `results/high_priority_bridges.csv` | Top 20 bridges by criticality |
   | `models/temporal_gnn_bridge_hi_delta_monotonic.pt` | Trained model |

The notebook always trains a new model. The checkpoints in `models/` are not reloaded and only fit the dataset they were trained on.

## Step 4: Build the dashboard data

From the repository root:

```bash
python prepare_dashboard_data.py
```

This reads `data/my_state_merged.csv` and the criticality CSVs in `results/`, then replaces `dashboard_data.json` with your bridges. If your merged file has a different name, pass it in:

```bash
python prepare_dashboard_data.py --input data/another_file.csv
```

> **Before committing:** following these steps changes tracked files, which then show up in `git status`: `dashboard_data.json`, `models/temporal_gnn_bridge_hi_delta_monotonic.pt`, and `notebooks/forecasting_arranged.ipynb`. Leave them out of your commits unless you mean to publish them.

## Step 5: Open the dashboard

```bash
python serve_dashboard.py
```

This serves the repository on port 8000, reachable only from your own computer, and opens this page in your browser:

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

<!-- Screenshot: uncomment after adding the file.
![Comparing bridges](docs/images/compare.png)
-->

**Scenarios**
- Choose a preset (Baseline, Climate Stress, Heavy Traffic, Extreme Weather, Optimistic) or set your own values with the sliders.
- **Apply to Map** carries the scenario over to the map, and **Reset to Baseline** clears it. Scenarios change Bridge Avg HI and Worst Element HI, but not Criticality Score or the per-element values in the bridge panel.
- Scenarios are a quick estimate that scales the forecast's year-over-year HI drops in the browser. They do not re-run the model.

<!-- Screenshot or GIF: uncomment after adding the file.
![Scenario comparison](docs/images/scenarios.png)
-->

The dashboard has no export or download feature, and comparisons and scenarios are lost when the page reloads.

## Optional components

You don't need any of these to run the steps above.

### Forecast with uncertainty

`notebooks/forecasting_arranged_mc_dropout_updated.ipynb` extends the main notebook with Monte Carlo dropout and baseline comparisons. It uses `scripts/bridge_criticality_pipeline_updated.py`, which scores criticality from 0 to 100. The dashboard converts that scale automatically. This notebook overwrites the main notebook's result CSVs and model checkpoint, so Step 4 uses whichever notebook you ran last.

`notebooks/forecasting.ipynb` is an early baseline kept for reference. It does not run from top to bottom.

### Neo4j and Gemini settings

The static graph model and the Streamlit dashboard read their settings from a `.env` file in the repository root. Git ignores this file, so your credentials stay on your machine. Create it from the template, then fill in your own values:

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

| Setting | Used for |
|---|---|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Your Neo4j connection |
| `GOOGLE_API_KEY` | The Gemini AI assistant in the Streamlit dashboard |

### Static graph model

This separate track trains a bridge-level graph model from a Neo4j database. It needs a running Neo4j instance, the `.env` file above, and these extra packages:

```bash
pip install neo4j rdflib rdflib-neo4j python-dotenv
```

> **Warning:** the code under **Newer version w/ neo4j compatability** in `notebooks/ontologyBuilder.ipynb` runs `MATCH (n) DETACH DELETE n`, which deletes everything in the connected database. Use a dedicated Neo4j instance.

1. Start Jupyter from `notebooks/` and open `ontologyBuilder.ipynb`. If your merged file is not `data/my_state_merged.csv`, update `csv_file_path` under **Data population**.
2. Run the cells under **Newer version w/ neo4j compatability** and **Data population**, in order.
3. From the repository root, run:

   ```bash
   python scripts/no_torch.py
   python scripts/bridge_extract_toTorch.py
   python scripts/train_bridge_gnn.py
   ```

### Streamlit dashboard

`scripts/dashboard.py` is a separate app for exploring the Neo4j graph, with an AI assistant powered by Gemini. It does not show the forecasts. Create the `.env` file described above, then run:

```bash
pip install streamlit pydeck plotly neo4j google-generativeai python-dotenv
streamlit run scripts/dashboard.py
```

## How the scores work

**Forecast model.** A graph convolutional network combined with a GRU learns how element health changes from year to year, using a graph that links each bridge to its elements. It trains on the 2020→2021, 2021→2022, and 2022→2023 transitions and is tested on 2023→2024. Predicted HI can only stay level or fall, and each yearly drop is limited to 15%.

**Climate-Traffic Impact Matrix (CTIM), 0–1.** Each hazard is scaled to an exposure `E` from 0 to 1, then combined with an expert-set vulnerability matrix `V` of element types and hazards:

```
element impact = sum(V[element, hazard] * E[hazard]^2) / sum(E[hazard])
CTIM           = average element impact across the 8 element types
```

**Criticality score, 0–1.** Higher means more urgent:

```
criticality          = 0.45 * HI penalty + 0.35 * CTIM + 0.20 * network significance
HI penalty           = 1 - HI / 100
network significance = 0.5 * min(ADT / 120000, 1) + 0.3 * min(detour km / 100, 1) + 0.2 if on an essential route
```

The notebook currently leaves detour length at 0 and essential route off. The hazards, their value ranges, and the vulnerability matrix are defined in [notebooks/bridge_criticality_pipeline.py](notebooks/bridge_criticality_pipeline.py). The uncertainty notebook's updated pipeline uses different weights (40/40/20) and a 0–100 scale.

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
| `FileNotFoundError` from `prepare_dashboard_data.py` | Run the notebook first. If your merged file has a different name, use `--input` as shown in [Step 4](#step-4-build-the-dashboard-data) |
| No bridges in `dashboard_data.json` | Confirm the latitude and longitude columns are filled in, and that the merged CSV is the one you ran the notebook on |
| "Failed to load dashboard data" | Start the page with `python serve_dashboard.py`, confirm `dashboard_data.json` exists, and check your internet connection |
| Port 8000 is already in use | Run `python -m http.server 8001 --bind 127.0.0.1`, then open `http://localhost:8001/bridge_criticality_interactive_dashboard.html` |
| `KeyError: 'NEO4J_PASSWORD'` | Create the `.env` file described in [Neo4j and Gemini settings](#neo4j-and-gemini-settings) |
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
| `.env.example` | Template for your Neo4j and Gemini settings |

## Resources

- [Literature review progress](https://docs.google.com/spreadsheets/d/1FBF4gexEheqrDxaIwI9gxIBC1Uaw7EUmpFTdzaxkVrw/edit?usp=sharing): reviewed papers, key findings, and connections to this project.
- [FHWA Bridge Management Guide (PDF)](https://www.fhwa.dot.gov/bridge/mtguide.pdf): U.S. Federal Highway Administration manual on bridge management practices.
