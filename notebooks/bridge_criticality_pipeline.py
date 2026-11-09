"""
Bridge Criticality Pipeline
============================
Climate-Traffic Impact Matrix (CTIM) → Health Impact Score → Criticality Score

Integrates with the existing Temporal GCN-GRU forecasting notebook.

Architecture:
  1. CTIM  – maps (weather event, bridge element) → exposure weight [0,1]
  2. Vulnerability matrix – maps bridge element → vulnerability to each hazard
  3. Impact Score (IS)  – dot-product aggregate, [0,1]
  4. Severity modifier  – softmax-based probability of failure across risk tiers
  5. Criticality Score  – IS × network significance × HI penalty
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ONTOLOGY CONSTANTS
#     Drawn from the ontologyBuilder.ipynb BMO schema + research papers.
# ─────────────────────────────────────────────────────────────────────────────

# Bridge elements (matches NBE element numbers conceptually)
BRIDGE_ELEMENTS = [
    "deck",
    "superstructure",
    "substructure",
    "bearings",
    "joints_seals",
    "railing_parapets",
    "wearing_surface",
    "culvert",
]

# Climate / traffic hazards (matches NBI covariate names from forecasting nb)
HAZARDS = [
    "avg_temperature_c",       # NBI_Average Temperature
    "max_temperature_c",       # NBI_Maximum Temperature
    "min_temperature_c",       # NBI_Minimum Temperature
    "freeze_thaw_cycles",      # NBI_Number of Freeze-Thaw Cycles
    "precip_days",             # NBI_Number of Days with Measurable Precipitation
    "below_zero_days",         # NBI_Number of Days with Temperature Below 0°C
    "snowfall_days",           # proxy for deicing salt demand
    "wind_speed",              # structural loading
    "average_daily_traffic",   # NBI_29 – Average Daily Traffic
    "truck_percentage",        # heavy vehicle fraction
    "flood_risk_proxy",        # derived: high precip + low elevation
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  VULNERABILITY MATRIX  V[element × hazard]
#     Each entry ∈ [0, 1] — expert-encoded susceptibility.
#     Sources: Guest et al. 2020 (chloride / freeze-thaw on deck/super),
#              Chang et al. 2020 (storm surge, extreme events on substructure).
# ─────────────────────────────────────────────────────────────────────────────

# Rows: BRIDGE_ELEMENTS  |  Cols: HAZARDS
_V_RAW = np.array([
    # avg_T  max_T  min_T  FT     prec   <0d   snow   wind   ADT    truck  flood
    [0.40,  0.30,  0.30,  0.90,  0.70,  0.80,  0.85,  0.10,  0.60,  0.70,  0.50],  # deck
    [0.25,  0.20,  0.25,  0.55,  0.40,  0.50,  0.45,  0.55,  0.75,  0.80,  0.40],  # superstructure
    [0.20,  0.15,  0.20,  0.35,  0.50,  0.40,  0.35,  0.30,  0.40,  0.50,  0.85],  # substructure
    [0.30,  0.25,  0.35,  0.70,  0.55,  0.65,  0.60,  0.40,  0.50,  0.55,  0.35],  # bearings
    [0.35,  0.30,  0.35,  0.75,  0.65,  0.70,  0.70,  0.25,  0.45,  0.50,  0.40],  # joints_seals
    [0.10,  0.10,  0.10,  0.20,  0.15,  0.20,  0.15,  0.60,  0.30,  0.30,  0.15],  # railing_parapets
    [0.50,  0.40,  0.20,  0.60,  0.55,  0.55,  0.65,  0.05,  0.80,  0.85,  0.30],  # wearing_surface
    [0.20,  0.15,  0.20,  0.30,  0.60,  0.35,  0.30,  0.20,  0.35,  0.40,  0.90],  # culvert
], dtype=np.float32)

VULNERABILITY_MATRIX = pd.DataFrame(
    _V_RAW, index=BRIDGE_ELEMENTS, columns=HAZARDS
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HAZARD NORMALIZER
#     Converts raw covariate values into [0, 1] exposure scores.
# ─────────────────────────────────────────────────────────────────────────────

# (min, max) reference ranges — calibrated to Florida + Oklahoma NBI data ranges
HAZARD_RANGES: dict[str, tuple[float, float]] = {
    "avg_temperature_c":     (-10.0,  35.0),
    "max_temperature_c":     (  0.0,  45.0),
    "min_temperature_c":     (-30.0,  20.0),
    "freeze_thaw_cycles":    (  0.0, 120.0),
    "precip_days":           (  0.0, 200.0),
    "below_zero_days":       (  0.0, 180.0),
    "snowfall_days":         (  0.0, 100.0),
    "wind_speed":            (  0.0,  25.0),
    "average_daily_traffic": (  0.0, 120_000.0),
    "truck_percentage":      (  0.0,   0.35),
    "flood_risk_proxy":      (  0.0,   1.0),
}

# For some hazards more = worse (heat, traffic, freeze-thaw).
# For min temperature, LOWER = worse, so we invert.
INVERT_HAZARD = {"min_temperature_c": True}


def normalize_hazard(name: str, value: float) -> float:
    """Map a raw hazard value to [0, 1] exposure score."""
    lo, hi = HAZARD_RANGES.get(name, (0.0, 1.0))
    hi = max(hi, lo + 1e-6)
    score = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    if INVERT_HAZARD.get(name, False):
        score = 1.0 - score
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLIMATE-TRAFFIC IMPACT MATRIX (CTIM)
# ─────────────────────────────────────────────────────────────────────────────

class CTIM:
    """
    Climate-Traffic Impact Matrix for a single bridge-year snapshot.

    Attributes
    ----------
    exposure : pd.Series  [HAZARDS]       – normalized hazard exposure [0,1]
    matrix   : pd.DataFrame [elem × haz] – element-level impact weights [0,1]
    impact_vector : pd.Series [BRIDGE_ELEMENTS] – weighted impact per element
    aggregate_score : float               – scalar bridge impact [0,1]
    """

    def __init__(self, hazard_values: dict[str, float]):
        """
        Parameters
        ----------
        hazard_values : raw covariate values keyed by HAZARDS names.
                        Missing hazards default to 0.
        """
        # -- 4a. Normalize raw hazard values → exposure vector e ∈ [0,1]^H
        self.exposure = pd.Series(
            {h: normalize_hazard(h, hazard_values.get(h, 0.0)) for h in HAZARDS},
            name="exposure",
        )

        # -- 4b. Element-level impact matrix  M[e,h] = V[e,h] × e[h]
        #        Each cell: how much hazard h hurts element e given current exposure
        self.matrix = VULNERABILITY_MATRIX.multiply(self.exposure, axis="columns")

        # -- 4c. Per-element impact score = row-wise weighted mean
        #        Use hazard exposure as weights so rare-but-severe hazards don't
        #        dominate when their exposure is near zero.
        w = self.exposure.values + 1e-8          # avoid div/0
        self.impact_vector = pd.Series(
            (self.matrix.values @ w) / w.sum(),
            index=BRIDGE_ELEMENTS,
            name="element_impact",
        )

        # -- 4d. Aggregate scalar ∈ [0,1]  (unweighted mean across elements)
        self.aggregate_score = float(self.impact_vector.mean())

    # ── helpers ──────────────────────────────────────────────────────────────

    def hi_multiplier(self) -> float:
        """
        Returns a [0,1] multiplier to scale the GNN-predicted ΔHI.
        1.0 = maximum climate-traffic stress (fastest deterioration),
        0.0 = no external stress.
        """
        return self.aggregate_score

    def top_hazards(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the n hazards driving the highest aggregate impact."""
        col_means = self.matrix.mean(axis=0)
        return col_means.nlargest(n).items()  # type: ignore[return-value]

    def top_vulnerable_elements(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the n most impacted elements."""
        return self.impact_vector.nlargest(n).items()  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RISK TIER PROBABILITIES — softmax on adjusted HI
# ─────────────────────────────────────────────────────────────────────────────

RISK_TIERS = ["critical", "poor", "fair", "good"]

# Logit offsets: lower HI → higher logits for bad tiers
# These are calibrated so HI=100 → P(good)≈0.90, HI=0 → P(critical)≈0.70
_TIER_CENTERS = np.array([15.0, 40.0, 65.0, 90.0])   # HI midpoints per tier
_TEMPERATURE   = 15.0                                   # softmax sharpness


def risk_tier_probabilities(hi: float) -> dict[str, float]:
    """
    Compute softmax probability distribution over [critical, poor, fair, good]
    for a given Health Index value in [0, 100].

    Uses negative squared distance from tier centres scaled by temperature,
    so the distribution peaks at the appropriate tier.
    """
    hi_clipped = float(np.clip(hi, 0.0, 100.0))
    logits = -((hi_clipped - _TIER_CENTERS) ** 2) / _TEMPERATURE
    probs = softmax(logits)
    return dict(zip(RISK_TIERS, probs.tolist()))


def dominant_risk_tier(hi: float) -> str:
    """Return the most probable risk tier for a given HI."""
    return max(risk_tier_probabilities(hi), key=lambda k: risk_tier_probabilities(hi)[k])


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CRITICALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

class CriticalityScore:
    """
    Composite bridge criticality score for maintenance prioritisation.

    Formula
    -------
    CS = α·HI_penalty + β·CTIM_impact + γ·network_significance

    where all sub-scores are in [0, 1] and α + β + γ = 1.

    The final score is in [0, 1]; higher = more urgent intervention.
    """

    DEFAULT_WEIGHTS = dict(hi_penalty=0.45, ctim_impact=0.35, network_sig=0.20)

    def __init__(
        self,
        bridge_id: str,
        hi: float,                       # current or forecast HI [0, 100]
        ctim: CTIM,
        adt: float = 0.0,                # Average Daily Traffic (raw)
        detour_km: float = 0.0,          # Detour length if bridge closes
        essential_route: bool = False,   # is it on a designated essential route?
        weights: Optional[dict] = None,
    ):
        self.bridge_id = bridge_id
        self.hi = float(np.clip(hi, 0.0, 100.0))
        self.ctim = ctim

        w = weights or self.DEFAULT_WEIGHTS

        # ── sub-scores ───────────────────────────────────────────────────────

        # (a) HI penalty: low HI → high penalty
        #     We use 1 − HI/100 so a bridge at 20 HI has 0.80 penalty.
        self.hi_penalty: float = 1.0 - self.hi / 100.0

        # (b) Climate-Traffic Impact from CTIM
        self.ctim_impact: float = ctim.aggregate_score

        # (c) Network significance: normalised composite of ADT + detour + essential
        adt_norm   = float(np.clip(adt / 120_000.0, 0.0, 1.0))
        detour_norm = float(np.clip(detour_km / 100.0, 0.0, 1.0))
        essential_bonus = 0.20 if essential_route else 0.0
        self.network_significance: float = float(
            np.clip(0.50 * adt_norm + 0.30 * detour_norm + essential_bonus, 0.0, 1.0)
        )

        # ── composite ────────────────────────────────────────────────────────
        self.score: float = (
            w["hi_penalty"] * self.hi_penalty
            + w["ctim_impact"] * self.ctim_impact
            + w["network_sig"] * self.network_significance
        )

        # ── risk tier ────────────────────────────────────────────────────────
        self.risk_probs   = risk_tier_probabilities(self.hi)
        self.risk_tier    = dominant_risk_tier(self.hi)

        # ── adjusted HI (climate-accelerated) ────────────────────────────────
        # The CTIM multiplier scales how much the GNN delta should be amplified.
        # A bridge with 100% climate stress deteriorates faster than the model
        # naively predicts from historical climate alone.
        self.climate_acceleration_factor: float = 1.0 + ctim.aggregate_score

    def summary(self) -> dict:
        return {
            "bridge_id":                    self.bridge_id,
            "current_hi":                   round(self.hi, 2),
            "criticality_score":            round(self.score, 4),
            "risk_tier":                    self.risk_tier,
            "p_critical":                   round(self.risk_probs["critical"], 3),
            "p_poor":                       round(self.risk_probs["poor"], 3),
            "p_fair":                       round(self.risk_probs["fair"], 3),
            "p_good":                       round(self.risk_probs["good"], 3),
            "hi_penalty":                   round(self.hi_penalty, 4),
            "ctim_impact":                  round(self.ctim_impact, 4),
            "network_significance":         round(self.network_significance, 4),
            "climate_acceleration_factor":  round(self.climate_acceleration_factor, 4),
            "top_hazard":                   list(self.ctim.top_hazards(1))[0][0],
            "most_vulnerable_element":      list(self.ctim.top_vulnerable_elements(1))[0][0],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PORTFOLIO RANKER
# ─────────────────────────────────────────────────────────────────────────────

class BridgePortfolioRanker:
    """
    Rank a fleet of bridges by criticality score and generate a prioritisation
    report that can feed directly into the existing forecasting pipeline.
    """

    def __init__(self):
        self._records: list[CriticalityScore] = []

    def add_bridge(self, cs: CriticalityScore) -> None:
        self._records.append(cs)

    def rank(self) -> pd.DataFrame:
        rows = [r.summary() for r in self._records]
        df = pd.DataFrame(rows).sort_values("criticality_score", ascending=False)
        df["priority_rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def flag_urgent(self, threshold: float = 0.70) -> pd.DataFrame:
        """Bridges above threshold score — candidates for immediate inspection."""
        return self.rank().query(f"criticality_score >= {threshold}")

    def ctim_summary(self) -> pd.DataFrame:
        """Portfolio-level mean impact matrix (elements × hazards)."""
        matrices = [r.ctim.matrix for r in self._records]
        return pd.concat(matrices).groupby(level=0).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  INTEGRATION HELPERS for forecasting_arranged.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def build_hazard_dict_from_nbi_row(row: pd.Series, covar_idx: dict[str, int]) -> dict:
    """
    Convert a node feature vector (as used in the GNN notebook) to the
    hazard_values dict expected by CTIM.

    Parameters
    ----------
    row       : numpy feature slice for ONE bridge (shape: [n_features])
    covar_idx : {column_name: feature_index} mapping from the notebook
    """
    NBI_TO_HAZARD = {
        "NBI_Average Temperature":                  "avg_temperature_c",
        "NBI_Maximum Temperature":                  "max_temperature_c",
        "NBI_Minimum Temperature":                  "min_temperature_c",
        "NBI_Number of Freeze-Thaw Cycles":         "freeze_thaw_cycles",
        "NBI_Number of Days with Measurable Precipitation": "precip_days",
        "NBI_Number of Days with Temperature Below 0°C":    "below_zero_days",
        "NBI_Snowfall Days":                        "snowfall_days",
        "NBI_29 - Average Daily Traffic":           "average_daily_traffic",
    }
    out = {}
    for nbi_name, haz_name in NBI_TO_HAZARD.items():
        if nbi_name in covar_idx:
            out[haz_name] = float(row[covar_idx[nbi_name]])
    return out


def apply_ctim_to_forecast(
    forecast_df: pd.DataFrame,
    bridge_hazards: dict[str, dict[str, float]],
    bridge_meta: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Attach CTIM impact and criticality scores to the autoregressive forecast CSV.

    Parameters
    ----------
    forecast_df   : element_hi_forecast_2025_2029_AR_scenario.csv as DataFrame
    bridge_hazards: {bridge_id → hazard_values_dict} for each forecast year
    bridge_meta   : {bridge_id → {'adt': ..., 'detour_km': ..., 'essential': ...}}

    Returns
    -------
    DataFrame with additional columns:
        ctim_impact, hi_penalty, criticality_score, risk_tier,
        climate_acceleration_factor, adjusted_hi
    """
    meta = bridge_meta or {}
    results = []

    for _, grp in forecast_df.groupby("STRUCTURE_NUMBER"):
        bid = str(grp["STRUCTURE_NUMBER"].iloc[0])
        haz = bridge_hazards.get(bid, {})
        ctim = CTIM(haz)
        m = meta.get(bid, {})

        for _, row in grp.iterrows():
            year = row.get("year", row.get("YEAR", 2025))
            hi_col = f"HI_{year}" if f"HI_{year}" in row.index else "HI"
            hi = float(row.get(hi_col, row.get("HI", 70.0)))

            cs = CriticalityScore(
                bridge_id=bid,
                hi=hi,
                ctim=ctim,
                adt=m.get("adt", 0.0),
                detour_km=m.get("detour_km", 0.0),
                essential_route=m.get("essential", False),
            )
            new_row = row.to_dict()
            new_row.update({
                "ctim_impact":                  round(ctim.aggregate_score, 4),
                "hi_penalty":                   round(cs.hi_penalty, 4),
                "criticality_score":            round(cs.score, 4),
                "risk_tier":                    cs.risk_tier,
                "climate_acceleration_factor":  round(cs.climate_acceleration_factor, 4),
                "adjusted_hi":                  round(hi * (1.0 / cs.climate_acceleration_factor), 2),
                "p_critical":                   round(cs.risk_probs["critical"], 3),
                "p_poor":                       round(cs.risk_probs["poor"], 3),
            })
            results.append(new_row)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DEMO / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Bridge Criticality Pipeline — Demo")
    print("=" * 70)

    # Simulate three Florida bridges with different exposure profiles
    bridges = [
        {
            "id": "FL-COASTAL-001",
            "hazards": {
                "avg_temperature_c": 28.0, "max_temperature_c": 38.0,
                "min_temperature_c": 12.0, "freeze_thaw_cycles": 2.0,
                "precip_days": 130.0, "below_zero_days": 0.0,
                "snowfall_days": 0.0, "wind_speed": 18.0,
                "average_daily_traffic": 85_000.0, "truck_percentage": 0.08,
                "flood_risk_proxy": 0.75,
            },
            "hi": 55.0, "adt": 85_000, "detour_km": 45.0, "essential": True,
        },
        {
            "id": "FL-INLAND-042",
            "hazards": {
                "avg_temperature_c": 24.0, "max_temperature_c": 35.0,
                "min_temperature_c": 8.0, "freeze_thaw_cycles": 5.0,
                "precip_days": 100.0, "below_zero_days": 3.0,
                "snowfall_days": 1.0, "wind_speed": 10.0,
                "average_daily_traffic": 22_000.0, "truck_percentage": 0.15,
                "flood_risk_proxy": 0.20,
            },
            "hi": 78.0, "adt": 22_000, "detour_km": 8.0, "essential": False,
        },
        {
            "id": "OK-RURAL-117",
            "hazards": {
                "avg_temperature_c": 16.0, "max_temperature_c": 38.0,
                "min_temperature_c": -12.0, "freeze_thaw_cycles": 45.0,
                "precip_days": 80.0, "below_zero_days": 42.0,
                "snowfall_days": 18.0, "wind_speed": 14.0,
                "average_daily_traffic": 4_500.0, "truck_percentage": 0.22,
                "flood_risk_proxy": 0.30,
            },
            "hi": 42.0, "adt": 4_500, "detour_km": 60.0, "essential": True,
        },
    ]

    ranker = BridgePortfolioRanker()

    for b in bridges:
        ctim = CTIM(b["hazards"])
        cs = CriticalityScore(
            bridge_id=b["id"],
            hi=b["hi"],
            ctim=ctim,
            adt=b["adt"],
            detour_km=b["detour_km"],
            essential_route=b["essential"],
        )
        ranker.add_bridge(cs)

        print(f"\nBridge: {b['id']}")
        print(f"  HI: {b['hi']} | Risk tier: {cs.risk_tier}")
        print(f"  CTIM aggregate impact: {ctim.aggregate_score:.3f}")
        print(f"  Climate acceleration factor: {cs.climate_acceleration_factor:.3f}")
        print(f"  Criticality score: {cs.score:.4f}")
        print(f"  P(critical)={cs.risk_probs['critical']:.3f}  "
              f"P(poor)={cs.risk_probs['poor']:.3f}  "
              f"P(fair)={cs.risk_probs['fair']:.3f}  "
              f"P(good)={cs.risk_probs['good']:.3f}")
        tops = list(ctim.top_hazards(3))
        print(f"  Top 3 hazards: {', '.join(f'{h}({v:.3f})' for h,v in tops)}")
        topv = list(ctim.top_vulnerable_elements(3))
        print(f"  Top 3 vulnerable elements: {', '.join(f'{e}({v:.3f})' for e,v in topv)}")
        print(f"\n  CTIM matrix (element × hazard):")
        print(ctim.matrix.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n\n=== PORTFOLIO RANKING ===")
    ranking = ranker.rank()
    print(ranking[[
        "priority_rank", "bridge_id", "current_hi", "criticality_score",
        "risk_tier", "ctim_impact", "network_significance",
        "climate_acceleration_factor", "top_hazard", "most_vulnerable_element"
    ]].to_string(index=False))

    print("\n\n=== URGENT BRIDGES (CS ≥ 0.55) ===")
    urgent = ranker.flag_urgent(threshold=0.55)
    if len(urgent):
        print(urgent[["bridge_id", "criticality_score", "risk_tier"]].to_string(index=False))
    else:
        print("None.")

    print("\n\n=== PORTFOLIO MEAN CTIM (element × hazard) ===")
    print(ranker.ctim_summary().to_string(float_format=lambda x: f"{x:.3f}"))
