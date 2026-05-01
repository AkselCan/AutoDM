"""
Bridge Criticality Pipeline  v2
=================================
Climate-Traffic Impact Matrix (CTIM) → Criticality Score → Priority Ranking

Changes from v1
---------------
  • CS weights rebalanced        → HI 40 / CTIM 40 / Network 20
  • HI penalty uses convex form  → (1 − HI/100)² punishes low HI harder
  • Softmax temperature lowered  → 80 (sharper tier separation)
  • All scores on 0–100 scale    → CS × 100 before reporting
  • Data-driven vulnerability    → LearnedVulnerabilityMatrix fits Ridge
                                   regression per element from historical HI
                                   deltas; falls back to expert priors when
                                   data are sparse; expert/learned blend param
  • Hazard interaction terms     → λ(FT × Truck) + γ(Precip × Flood) added
                                   to element impact formula
  • Essential route from data    → bridge is essential if ADT ≥ 75th pct
                                   of portfolio ADT distribution
  • Detour score via neighbours  → compute_detour_score() uses a simple
                                   neighbour table (bridge_id, distance_km,
                                   neighbour_adt) to assess real alternatives
  • Monte Carlo dropout removed  → uncertainty through CTIM spread only
  • Regression metrics kept      → weighted MAE, R² still available
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ONTOLOGY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BRIDGE_ELEMENTS = [
    "deck", "superstructure", "substructure", "bearings",
    "joints_seals", "railing_parapets", "wearing_surface", "culvert",
]

HAZARDS = [
    "avg_temperature_c", "max_temperature_c", "min_temperature_c",
    "freeze_thaw_cycles", "precip_days", "below_zero_days", "snowfall_days",
    "wind_speed", "average_daily_traffic", "truck_percentage", "flood_risk_proxy",
]

# Synergistic interaction terms: (hazard_a, hazard_b, weight λ)
# λ(FT × Truck): deicing salt demand amplified by heavy axle loading → deck spalling
# γ(Precip × Flood): wet season + flood-prone site → scour / substructure damage
INTERACTION_TERMS: list[tuple[str, str, float]] = [
    ("freeze_thaw_cycles", "truck_percentage",  0.15),
    ("precip_days",        "flood_risk_proxy",  0.12),
]

# Expert prior vulnerability matrix  V[element × hazard]
# Sources: Guest et al. 2020, Chang et al. 2020
_V_EXPERT = np.array([
    # avgT   maxT   minT   FT     prec   <0d    snow   wind   ADT    truck  flood
    [0.40,  0.30,  0.30,  0.90,  0.70,  0.80,  0.85,  0.10,  0.60,  0.70,  0.50],
    [0.25,  0.20,  0.25,  0.55,  0.40,  0.50,  0.45,  0.55,  0.75,  0.80,  0.40],
    [0.20,  0.15,  0.20,  0.35,  0.50,  0.40,  0.35,  0.30,  0.40,  0.50,  0.85],
    [0.30,  0.25,  0.35,  0.70,  0.55,  0.65,  0.60,  0.40,  0.50,  0.55,  0.35],
    [0.35,  0.30,  0.35,  0.75,  0.65,  0.70,  0.70,  0.25,  0.45,  0.50,  0.40],
    [0.10,  0.10,  0.10,  0.20,  0.15,  0.20,  0.15,  0.60,  0.30,  0.30,  0.15],
    [0.50,  0.40,  0.20,  0.60,  0.55,  0.55,  0.65,  0.05,  0.80,  0.85,  0.30],
    [0.20,  0.15,  0.20,  0.30,  0.60,  0.35,  0.30,  0.20,  0.35,  0.40,  0.90],
], dtype=np.float32)

VULNERABILITY_MATRIX_EXPERT = pd.DataFrame(
    _V_EXPERT, index=BRIDGE_ELEMENTS, columns=HAZARDS
)

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
INVERT_HAZARD = {"min_temperature_c": True}

ELEMENT_ALIASES = {
    "deck":             ["deck", "slab"],
    "superstructure":   ["superstructure", "girder", "beam", "truss"],
    "substructure":     ["substructure", "pier", "abutment", "foundation", "bent", "column", "footing"],
    "bearings":         ["bearing", "bearings"],
    "joints_seals":     ["joint", "joints", "seal", "seals"],
    "railing_parapets": ["railing", "parapet", "rail"],
    "wearing_surface":  ["wearing", "overlay", "surface", "coating", "membrane"],
    "culvert":          ["culvert", "pipe", "barrel"],
}

ELEMENT_NUMBER_MAP: dict[int, str] = {
    12: "deck",
    13: "wearing_surface", 14: "wearing_surface",
    15: "wearing_surface", 16: "wearing_surface",
    31: "superstructure",  32: "superstructure", 33: "superstructure",
    41: "substructure",    42: "substructure",   43: "substructure",
    47: "bearings",
    51: "joints_seals",
    60: "railing_parapets", 61: "railing_parapets",
    71: "culvert", 72: "culvert",
}

# Risk tier configuration
RISK_TIERS = ["critical", "poor", "fair", "good"]
_TIER_CENTERS = np.array([15.0, 40.0, 65.0, 90.0])
# Temperature=80: clear separation without collapsing to one-hot at boundaries
_SOFTMAX_TEMPERATURE = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_hazard(name: str, value: float) -> float:
    lo, hi = HAZARD_RANGES.get(name, (0.0, 1.0))
    hi = max(hi, lo + 1e-6)
    score = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return 1.0 - score if INVERT_HAZARD.get(name, False) else score


def normalize_element_name(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())
    for canonical, aliases in ELEMENT_ALIASES.items():
        if any(a in s for a in aliases):
            return canonical
    return None


def infer_element_key(row: pd.Series) -> str:
    for col in ["ELEMENT_KEY", "ELEMENT_NAME", "PARENT_ELEMENT_NAME", "ELEMENT_CATEGORY"]:
        if col in row.index:
            m = normalize_element_name(row.get(col))
            if m:
                return m
    for col in ["ELEMENT_NUMBER", "PARENT_ELEMENT_NUMBER"]:
        if col in row.index:
            try:
                num = int(float(row.get(col)))
                if num in ELEMENT_NUMBER_MAP:
                    return ELEMENT_NUMBER_MAP[num]
            except Exception:
                pass
    return "superstructure"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA-DRIVEN VULNERABILITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

class LearnedVulnerabilityMatrix:
    """
    Learns V[element, hazard] from historical NBI/NBE records via Ridge regression.

    For each element type, annual HI *decrease* (deterioration ≥ 0) is regressed
    on normalised hazard exposures. Fitted coefficients (clipped to [0,1]) are
    blended with the expert prior:

        V_blended = blend × V_expert + (1 − blend) × V_learned

    When n_samples < min_samples the expert prior is used unchanged for that row.

    Parameters
    ----------
    min_samples : int    Minimum rows per element before trusting the data.
    alpha       : float  Ridge regularisation strength.
    blend       : float  Expert prior weight in [0,1]. 0 = full data, 1 = full prior.
    """

    def __init__(self, min_samples: int = 30, alpha: float = 1.0, blend: float = 0.3):
        self.min_samples = min_samples
        self.alpha = alpha
        self.blend = blend
        self.matrix: pd.DataFrame = VULNERABILITY_MATRIX_EXPERT.copy()
        self._fitted: dict[str, bool] = {e: False for e in BRIDGE_ELEMENTS}
        self._n: dict[str, int] = {e: 0 for e in BRIDGE_ELEMENTS}

    def fit(self, df: pd.DataFrame) -> "LearnedVulnerabilityMatrix":
        """
        Fit from a historical DataFrame.

        Required columns
        ----------------
        element_key : one of BRIDGE_ELEMENTS
        hi_delta    : HI(t+1) − HI(t)  (negative = deterioration)
        + one column per HAZARD, already normalised to [0,1]
        """
        if df is None or len(df) == 0:
            return self

        if "hi_delta" not in df.columns:
            raise ValueError("DataFrame must contain 'hi_delta' = HI(t+1) - HI(t)")

        df = df.copy()
        df["_deterioration"] = (-df["hi_delta"]).clip(lower=0)

        learned_rows: dict[str, np.ndarray] = {}

        for elem in BRIDGE_ELEMENTS:
            sub = df[df["element_key"] == elem].dropna(subset=HAZARDS + ["_deterioration"])
            n = len(sub)
            self._n[elem] = n
            prior = _V_EXPERT[BRIDGE_ELEMENTS.index(elem)]

            if n < self.min_samples:
                learned_rows[elem] = prior.copy()
                self._fitted[elem] = False
                continue

            X = sub[HAZARDS].values.astype(float)
            y = sub["_deterioration"].values.astype(float)

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X, y)

            coefs = np.clip(model.coef_, 0.0, None)
            if coefs.max() > 0:
                coefs = coefs / coefs.max()   # normalise within element to [0,1]

            blended = np.clip(
                self.blend * prior + (1.0 - self.blend) * coefs, 0.0, 1.0
            ).astype(np.float32)
            learned_rows[elem] = blended
            self._fitted[elem] = True

        self.matrix = pd.DataFrame(
            np.vstack([learned_rows[e] for e in BRIDGE_ELEMENTS]),
            index=BRIDGE_ELEMENTS,
            columns=HAZARDS,
        )
        return self

    def is_data_driven(self, elem: str) -> bool:
        return self._fitted.get(elem, False)

    def coverage_report(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "element":     e,
            "n_samples":   self._n.get(e, 0),
            "data_driven": self._fitted.get(e, False),
        } for e in BRIDGE_ELEMENTS])


def fit_vulnerability_matrix(
    df: pd.DataFrame,
    min_samples: int = 30,
    alpha: float = 1.0,
    blend: float = 0.3,
) -> LearnedVulnerabilityMatrix:
    """Convenience wrapper. Call once with your historical NBI/NBE DataFrame."""
    return LearnedVulnerabilityMatrix(min_samples=min_samples, alpha=alpha, blend=blend).fit(df)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLIMATE-TRAFFIC IMPACT MATRIX  (CTIM)
# ─────────────────────────────────────────────────────────────────────────────

class CTIM:
    """
    Element × hazard impact matrix for one bridge-year snapshot.

    Extended element impact formula:
        Impact(e) = Σ_h V[e,h]·Exp(h)
                  + λ · Exp(FT) · Exp(Truck)      [FT–truck synergy]
                  + γ · Exp(Precip) · Exp(Flood)  [Precip–flood synergy]

    All values in [0, 1].
    """

    def __init__(
        self,
        hazard_values: dict[str, float],
        vuln: Optional[LearnedVulnerabilityMatrix] = None,
    ):
        V = vuln.matrix if vuln is not None else VULNERABILITY_MATRIX_EXPERT

        # ── normalised exposure vector
        self.exposure = pd.Series(
            {h: normalize_hazard(h, hazard_values.get(h, 0.0)) for h in HAZARDS},
            name="exposure",
        )

        # ── base impact matrix  M[e,h] = V[e,h] · Exp(h)
        self.matrix = V.multiply(self.exposure, axis="columns")

        # ── interaction bonuses
        self._interaction_bonus = self._compute_interactions()

        # ── per-element impact score (exposure-weighted mean + interaction)
        w = self.exposure.values + 1e-8
        base = (self.matrix.values @ w) / w.sum()
        bonus_arr = np.array([self._interaction_bonus.get(e, 0.0) for e in BRIDGE_ELEMENTS])
        self.impact_vector = pd.Series(
            np.clip(base + bonus_arr, 0.0, 1.0),
            index=BRIDGE_ELEMENTS,
            name="element_impact",
        )
        self.aggregate_score = float(self.impact_vector.mean())

    def _compute_interactions(self) -> dict[str, float]:
        exp = self.exposure
        bonus = {e: 0.0 for e in BRIDGE_ELEMENTS}
        for haz_a, haz_b, lam in INTERACTION_TERMS:
            if haz_a not in exp.index or haz_b not in exp.index:
                continue
            strength = lam * exp[haz_a] * exp[haz_b]
            if strength < 1e-6:
                continue
            for elem in BRIDGE_ELEMENTS:
                v_a = float(VULNERABILITY_MATRIX_EXPERT.loc[elem, haz_a])
                v_b = float(VULNERABILITY_MATRIX_EXPERT.loc[elem, haz_b])
                bonus[elem] += strength * (v_a + v_b) / 2.0
        return bonus

    def get_element_impact(self, element_key: Optional[str]) -> float:
        if element_key in self.impact_vector.index:
            return float(self.impact_vector.loc[element_key])
        return self.aggregate_score

    def top_hazards(self, n: int = 3) -> list[tuple[str, float]]:
        return list(self.matrix.mean(axis=0).nlargest(n).items())

    def top_vulnerable_elements(self, n: int = 3) -> list[tuple[str, float]]:
        return list(self.impact_vector.nlargest(n).items())

    def interaction_summary(self) -> dict[str, float]:
        return {e: round(v, 4) for e, v in self._interaction_bonus.items() if v > 1e-4}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RISK TIER PROBABILITIES
# ─────────────────────────────────────────────────────────────────────────────

def risk_tier_probabilities(hi: float) -> dict[str, float]:
    """
    Softmax distribution over [critical, poor, fair, good] for a given HI.
    Temperature = 80 → clear peak at the right tier, non-zero mass at neighbours.
    """
    hi_c = float(np.clip(hi, 0.0, 100.0))
    logits = -((hi_c - _TIER_CENTERS) ** 2) / _SOFTMAX_TEMPERATURE
    return dict(zip(RISK_TIERS, softmax(logits).tolist()))


def dominant_risk_tier(hi: float) -> str:
    p = risk_tier_probabilities(hi)
    return max(p, key=p.__getitem__)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  NETWORK SIGNIFICANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_adt_threshold(adt_series: pd.Series, percentile: float = 75.0) -> float:
    """75th-percentile ADT cutoff for 'essential' classification."""
    return float(np.nanpercentile(adt_series.dropna(), percentile))


def is_essential_from_data(adt: float, adt_threshold: float) -> bool:
    """True when this bridge is in the top-25 % of portfolio traffic."""
    return adt >= adt_threshold


def compute_detour_score(
    bridge_id: str,
    bridge_adt: float,
    neighbour_table: Optional[pd.DataFrame] = None,
    detour_km: float = 0.0,
) -> float:
    """
    Detour difficulty score ∈ [0, 1].  Higher = worse detour.

    Uses a lightweight neighbour table (no full KG required):
        columns: bridge_id, neighbour_id, distance_km, neighbour_adt

    Logic:
      - If a nearby bridge (≤ 20 km) carries ≥ 50 % of this bridge's ADT →
        viable alternative exists → lower detour penalty
      - No viable nearby bridge → high detour penalty
      - Falls back to normalised detour_km if no table supplied
    """
    detour_norm = float(np.clip(detour_km / 100.0, 0.0, 1.0))

    if neighbour_table is None or len(neighbour_table) == 0:
        return detour_norm

    nb = neighbour_table[neighbour_table["bridge_id"] == bridge_id]
    if nb.empty:
        return detour_norm

    near = nb[nb["distance_km"] <= 20.0]
    if near.empty:
        return float(np.clip(0.7 + 0.3 * detour_norm, 0.0, 1.0))

    best_adt = float(near["neighbour_adt"].max())
    if best_adt / max(bridge_adt, 1.0) >= 0.5:
        return float(np.clip(0.2 + 0.3 * detour_norm, 0.0, 1.0))
    else:
        return float(np.clip(0.5 + 0.4 * detour_norm, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CRITICALITY SCORE  (0–100 scale)
# ─────────────────────────────────────────────────────────────────────────────

class CriticalityScore:
    """
    Composite criticality score on a 0–100 scale.

    Formula
    -------
    raw_CS = 0.40·HI_penalty + 0.40·CTIM_impact + 0.20·network_significance
    CS     = raw_CS × 100

    HI_penalty = (1 − HI/100)²
        Convex shape: a bridge at HI=20 gets penalty 0.64,
        vs a linear penalty of 0.20 — 3.2× harder.
        This better reflects the exponential cost of very low HI bridges.
    """

    DEFAULT_WEIGHTS = dict(hi_penalty=0.40, ctim_impact=0.40, network_sig=0.20)

    def __init__(
        self,
        bridge_id: str,
        hi: float,
        ctim: CTIM,
        adt: float = 0.0,
        detour_score: float = 0.0,
        essential_route: bool = False,
        weights: Optional[dict] = None,
        element_key: Optional[str] = None,
        element_ctim_override: Optional[float] = None,
    ):
        self.bridge_id   = bridge_id
        self.hi          = float(np.clip(hi, 0.0, 100.0))
        self.ctim        = ctim
        self.element_key = element_key
        w = weights or self.DEFAULT_WEIGHTS

        # HI penalty — convex
        self.hi_penalty: float = (1.0 - self.hi / 100.0) ** 2

        # CTIM impact (element-specific when available)
        self.ctim_impact_bridge: float = ctim.aggregate_score
        self.ctim_impact: float = (
            element_ctim_override
            if element_ctim_override is not None
            else ctim.get_element_impact(element_key)
        )

        # Network significance
        adt_norm = float(np.clip(adt / 120_000.0, 0.0, 1.0))
        essential_bonus = 0.20 if essential_route else 0.0
        self.network_significance: float = float(
            np.clip(0.50 * adt_norm + 0.30 * detour_score + essential_bonus, 0.0, 1.0)
        )

        # Composite — raw [0,1] then scale to [0,100]
        raw = (
            w["hi_penalty"] * self.hi_penalty
            + w["ctim_impact"] * self.ctim_impact
            + w["network_sig"] * self.network_significance
        )
        self.score_raw: float = float(np.clip(raw, 0.0, 1.0))
        self.score: float = round(self.score_raw * 100.0, 2)

        # Risk tier
        self.risk_probs: dict[str, float] = risk_tier_probabilities(self.hi)
        self.risk_tier:  str              = dominant_risk_tier(self.hi)

        # CAF for GNN rollout
        self.climate_acceleration_factor: float = 1.0 + self.ctim_impact

    def summary(self) -> dict:
        top_h = list(self.ctim.top_hazards(1))[0]
        top_e = list(self.ctim.top_vulnerable_elements(1))[0]
        return {
            "bridge_id":                    self.bridge_id,
            "element_key":                  self.element_key or "—",
            "current_hi":                   round(self.hi, 2),
            "criticality_score":            self.score,
            "risk_tier":                    self.risk_tier,
            "p_critical":                   round(self.risk_probs["critical"], 3),
            "p_poor":                       round(self.risk_probs["poor"], 3),
            "p_fair":                       round(self.risk_probs["fair"], 3),
            "p_good":                       round(self.risk_probs["good"], 3),
            "hi_penalty":                   round(self.hi_penalty, 4),
            "ctim_impact":                  round(self.ctim_impact, 4),
            "ctim_impact_bridge":           round(self.ctim_impact_bridge, 4),
            "network_significance":         round(self.network_significance, 4),
            "climate_acceleration_factor":  round(self.climate_acceleration_factor, 4),
            "top_hazard":                   top_h[0],
            "most_vulnerable_element":      top_e[0],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PORTFOLIO RANKER
# ─────────────────────────────────────────────────────────────────────────────

class BridgePortfolioRanker:
    def __init__(self):
        self._records: list[CriticalityScore] = []

    def add_bridge(self, cs: CriticalityScore) -> None:
        self._records.append(cs)

    def rank(self) -> pd.DataFrame:
        rows = [r.summary() for r in self._records]
        df = pd.DataFrame(rows).sort_values("criticality_score", ascending=False)
        df["priority_rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def flag_urgent(self, threshold: float = 60.0) -> pd.DataFrame:
        """Bridges with CS ≥ threshold on 0–100 scale."""
        return self.rank().query(f"criticality_score >= {threshold}")

    def ctim_summary(self) -> pd.DataFrame:
        matrices = [r.ctim.matrix for r in self._records]
        return pd.concat(matrices).groupby(level=0).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  INTEGRATION HELPERS  (forecasting_arranged.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

NBI_TO_HAZARD: dict[str, str] = {
    "NBI_Average Temperature":                          "avg_temperature_c",
    "NBI_Maximum Temperature":                          "max_temperature_c",
    "NBI_Minimum Temperature":                          "min_temperature_c",
    "NBI_Number of Freeze-Thaw Cycles":                 "freeze_thaw_cycles",
    "NBI_Number of Days with Measurable Precipitation": "precip_days",
    "NBI_Number of Days with Temperature Below 0°C":    "below_zero_days",
    "NBI_Snowfall Days":                                "snowfall_days",
    "NBI_29 - Average Daily Traffic":                   "average_daily_traffic",
}


def build_hazard_dict_from_nbi_row(row: np.ndarray, covar_idx: dict[str, int]) -> dict:
    return {
        haz_name: float(row[covar_idx[nbi_name]])
        for nbi_name, haz_name in NBI_TO_HAZARD.items()
        if nbi_name in covar_idx
    }


def apply_ctim_to_forecast(
    forecast_df: pd.DataFrame,
    bridge_hazards: dict[str, dict[str, float]],
    bridge_meta: Optional[dict[str, dict]] = None,
    vuln: Optional[LearnedVulnerabilityMatrix] = None,
    adt_all: Optional[pd.Series] = None,
    neighbour_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach CTIM and criticality scores to the autoregressive forecast CSV.

    Parameters
    ----------
    forecast_df    : element_hi_forecast_2025_2029_AR_scenario.csv as DataFrame
    bridge_hazards : {bridge_id → hazard_values_dict}
    bridge_meta    : {bridge_id → {'adt': float, 'detour_km': float}}
    vuln           : fitted LearnedVulnerabilityMatrix (None → expert priors)
    adt_all        : all ADT values in portfolio (for essential threshold)
    neighbour_table: DataFrame [bridge_id, neighbour_id, distance_km, neighbour_adt]

    Returns
    -------
    DataFrame with criticality_score (0–100), risk_tier, ctim_impact,
    hi_penalty, climate_acceleration_factor, adjusted_hi, p_* columns.
    """
    meta = bridge_meta or {}

    # ADT threshold for essential flag
    if adt_all is not None:
        adt_thresh = compute_adt_threshold(adt_all)
    elif meta:
        adt_thresh = compute_adt_threshold(
            pd.Series([v.get("adt", 0.0) for v in meta.values()])
        )
    else:
        adt_thresh = 40_000.0

    results = []

    for _, grp in forecast_df.groupby("STRUCTURE_NUMBER"):
        bid       = str(grp["STRUCTURE_NUMBER"].iloc[0])
        haz       = bridge_hazards.get(bid, {})
        ctim      = CTIM(haz, vuln=vuln)
        m         = meta.get(bid, {})
        adt       = m.get("adt", haz.get("average_daily_traffic", 0.0))
        essential = is_essential_from_data(adt, adt_thresh)
        det_score = compute_detour_score(
            bid, adt, neighbour_table=neighbour_table, detour_km=m.get("detour_km", 0.0)
        )

        for _, row in grp.iterrows():
            year   = row.get("year", row.get("YEAR", 2025))
            hi_col = f"HI_{year}" if f"HI_{year}" in row.index else "HI"
            hi     = float(row.get(hi_col, row.get("HI", 70.0)))
            elem   = infer_element_key(row)
            elem_impact = ctim.get_element_impact(elem)

            cs = CriticalityScore(
                bridge_id=bid, hi=hi, ctim=ctim, adt=adt,
                detour_score=det_score, essential_route=essential,
                element_key=elem, element_ctim_override=elem_impact,
            )

            new_row = row.to_dict()
            new_row.update({
                "element_key":                  elem,
                "ctim_impact":                  round(elem_impact, 4),
                "ctim_impact_bridge":           round(ctim.aggregate_score, 4),
                "hi_penalty":                   round(cs.hi_penalty, 4),
                "network_significance":         round(cs.network_significance, 4),
                "criticality_score":            cs.score,
                "risk_tier":                    cs.risk_tier,
                "climate_acceleration_factor":  round(cs.climate_acceleration_factor, 4),
                "adjusted_hi":                  round(hi / cs.climate_acceleration_factor, 2),
                "p_critical":                   round(cs.risk_probs["critical"], 3),
                "p_poor":                       round(cs.risk_probs["poor"], 3),
                "p_fair":                       round(cs.risk_probs["fair"], 3),
                "p_good":                       round(cs.risk_probs["good"], 3),
                "is_essential":                 essential,
                "detour_score":                 round(det_score, 4),
            })
            results.append(new_row)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  REGRESSION METRICS  (MC dropout removed)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mae(
    y_true, y_pred,
    low_threshold: float = 70.0,
    critical_threshold: float = 50.0,
    low_weight: float = 2.0,
    critical_weight: float = 3.0,
) -> float:
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    weights = np.ones_like(y_true)
    weights[y_true < low_threshold]      = low_weight
    weights[y_true < critical_threshold] = critical_weight
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err    = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MAE":          float(np.mean(np.abs(err))),
        "Weighted_MAE": weighted_mae(y_true, y_pred),
        "RMSE":         float(np.sqrt(np.mean(err ** 2))),
        "R2":           float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11.  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("Bridge Criticality Pipeline  v2  —  smoke test")
    print("=" * 72)

    bridges = [
        {"id": "FL-COASTAL-001", "hi": 55.0, "adt": 85_000, "detour_km": 45.0,
         "hazards": {"avg_temperature_c": 28.0, "max_temperature_c": 38.0,
                     "min_temperature_c": 12.0, "freeze_thaw_cycles": 2.0,
                     "precip_days": 130.0, "below_zero_days": 0.0,
                     "snowfall_days": 0.0, "wind_speed": 18.0,
                     "average_daily_traffic": 85_000.0, "truck_percentage": 0.08,
                     "flood_risk_proxy": 0.75}},
        {"id": "FL-INLAND-042", "hi": 78.0, "adt": 22_000, "detour_km": 8.0,
         "hazards": {"avg_temperature_c": 24.0, "max_temperature_c": 35.0,
                     "min_temperature_c": 8.0, "freeze_thaw_cycles": 5.0,
                     "precip_days": 100.0, "below_zero_days": 3.0,
                     "snowfall_days": 1.0, "wind_speed": 10.0,
                     "average_daily_traffic": 22_000.0, "truck_percentage": 0.15,
                     "flood_risk_proxy": 0.20}},
        {"id": "OK-RURAL-117", "hi": 42.0, "adt": 4_500, "detour_km": 60.0,
         "hazards": {"avg_temperature_c": 16.0, "max_temperature_c": 38.0,
                     "min_temperature_c": -12.0, "freeze_thaw_cycles": 45.0,
                     "precip_days": 80.0, "below_zero_days": 42.0,
                     "snowfall_days": 18.0, "wind_speed": 14.0,
                     "average_daily_traffic": 4_500.0, "truck_percentage": 0.22,
                     "flood_risk_proxy": 0.30}},
    ]

    all_adts    = pd.Series([b["adt"] for b in bridges])
    adt_thresh  = compute_adt_threshold(all_adts)
    print(f"\nPortfolio ADT p75 threshold: {adt_thresh:,.0f}  → bridges above = essential\n")

    ranker = BridgePortfolioRanker()

    for b in bridges:
        ctim      = CTIM(b["hazards"])
        essential = is_essential_from_data(b["adt"], adt_thresh)
        det       = compute_detour_score(b["id"], b["adt"], detour_km=b["detour_km"])
        cs        = CriticalityScore(
            bridge_id=b["id"], hi=b["hi"], ctim=ctim,
            adt=b["adt"], detour_score=det, essential_route=essential,
        )
        ranker.add_bridge(cs)

        print(f"Bridge: {b['id']}")
        print(f"  HI={b['hi']}  essential={essential}  tier={cs.risk_tier}")
        print(f"  HI penalty: linear={1-b['hi']/100:.4f}  convex={cs.hi_penalty:.4f}")
        print(f"  CTIM aggregate : {ctim.aggregate_score:.4f}")
        print(f"  Interactions   : {ctim.interaction_summary()}")
        print(f"  Net. significance: {cs.network_significance:.4f}")
        print(f"  Criticality score: {cs.score:.1f} / 100")
        print(f"  Climate accel  : {cs.climate_acceleration_factor:.3f}×")
        p = cs.risk_probs
        print(f"  P(crit)={p['critical']:.3f} P(poor)={p['poor']:.3f} "
              f"P(fair)={p['fair']:.3f} P(good)={p['good']:.3f}\n")

    print("=== RANKING ===")
    print(ranker.rank()[[
        "priority_rank","bridge_id","current_hi","criticality_score",
        "risk_tier","hi_penalty","ctim_impact","network_significance",
    ]].to_string(index=False))

    print("\n=== HI PENALTY: convex vs linear ===")
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        lin = 1 - v/100; conv = lin**2
        print(f"  HI={v:3d}  linear={lin:.3f}  convex={conv:.4f}  ratio={conv/lin:.2f}×" if lin else f"  HI={v} linear=0")

    print("\n=== SOFTMAX TIERS (temperature=80) ===")
    for v in [10, 25, 42, 55, 65, 78, 92]:
        p = risk_tier_probabilities(v)
        print(f"  HI={v:3d}  crit={p['critical']:.3f}  poor={p['poor']:.3f}  "
              f"fair={p['fair']:.3f}  good={p['good']:.3f}  → {dominant_risk_tier(v)}")

    print("\n=== LEARNED VULNERABILITY MATRIX  (synthetic 300-row demo) ===")
    rng = np.random.default_rng(42)
    n   = 300
    syn = pd.DataFrame({
        "element_key": rng.choice(BRIDGE_ELEMENTS, n),
        "hi_delta":    rng.normal(-1.5, 3.0, n),
        **{h: rng.uniform(0, 1, n) for h in HAZARDS},
    })
    lvm = fit_vulnerability_matrix(syn, min_samples=20, blend=0.4)
    print(lvm.coverage_report().to_string(index=False))
    print("\nLearned deck row:")
    print(lvm.matrix.loc["deck"].round(3).to_string())