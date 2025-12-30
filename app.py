import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="CT ZIP Heatmap Platform", layout="wide")

# -----------------------
# Helpers
# -----------------------
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_zip_field(geojson: dict) -> str:
    # Common TIGER/Line ZCTA fields
    sample_props = geojson["features"][0].get("properties", {})
    for k in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"]:
        if k in sample_props:
            return k
    # fallback: try any property that looks like ZCTA
    for k in sample_props.keys():
        if "ZCTA" in k.upper():
            return k
    return ""  # unknown

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series([50] * len(s), index=s.index)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if mn == mx:
        return pd.Series([50] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100

# -----------------------
# Load files
# -----------------------
GEOJSON_PATH = "ct_zcta.json"  # your uploaded file
MODELS_PATH = "models.json"
OBESITY_PATH = "obesity_by_county.csv"
CROSSWALK_PATH = "zcta_to_county.csv"  # optional (you can add later)

if not os.path.exists(GEOJSON_PATH):
    st.error(f"Missing {GEOJSON_PATH} in repo root.")
    st.stop()

geo = load_geojson(GEOJSON_PATH)
zip_field = get_zip_field(geo)
if not zip_field:
    st.error("Could not detect ZIP field in GeoJSON properties. Expected ZCTA5CE20-like field.")
    st.stop()

# Create a base table of ZIPs from GeoJSON
zips = []
for feat in geo["features"]:
    props = feat.get("properties", {})
    z = str(props.get(zip_field, "")).strip()
    if z:
        zips.append(z)
base = pd.DataFrame({"zcta": sorted(set(zips))})

models = {}
if os.path.exists(MODELS_PATH):
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        models = json.load(f)

# -----------------------
# UI
# -----------------------
st.title("CT ZIP Heatmap Platform")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    model_names = list(models.keys()) if isinstance(models, dict) and models else [
        "Urgent Care (Premium)",
        "Gym (Premium)",
        "Gym (Budget)",
        "Fast Food (Budget)",
        "Fine Dining"
    ]
    chosen_model = st.selectbox("Business model", model_names, index=0)

with colB:
    animate = st.toggle("Animated reveal", value=True)

with colC:
    map_style = st.selectbox("Map style", ["carto-positron", "open-street-map"], index=0)

# -----------------------
# Build features (real if available, else placeholder)
# -----------------------
df = base.copy()

# Optional: apply obesity (county) if we have a crosswalk ZIP->county + obesity_by_county
has_crosswalk = os.path.exists(CROSSWALK_PATH)
has_obesity = os.path.exists(OBESITY_PATH)

if has_crosswalk:
    cw = safe_read_csv(CROSSWALK_PATH)
    # expected columns: zcta, county (case-insensitive)
    cols = {c.lower(): c for c in cw.columns}
    if "zcta" not in cols or "county" not in cols:
        st.warning("zcta_to_county.csv exists but must have columns: zcta, county")
    else:
        cw = cw.rename(columns={cols["zcta"]: "zcta", cols["county"]: "county"})
        cw["zcta"] = cw["zcta"].astype(str).str.zfill(5)
        df = df.merge(cw[["zcta", "county"]], on="zcta", how="left")

if has_obesity and "county" in df.columns:
    ob = safe_read_csv(OBESITY_PATH)
    # try to detect columns
    ob_cols = {c.lower(): c for c in ob.columns}
    # common possibilities
    county_col = ob_cols.get("county") or ob_cols.get("name") or list(ob.columns)[0]
    rate_col = None
    for k in ["obesity", "obesity_rate", "rate", "value", "percent", "pct"]:
        if k in ob_cols:
            rate_col = ob_cols[k]
            break
    if rate_col is None and len(ob.columns) >= 2:
        rate_col = list(ob.columns)[1]

    ob = ob.rename(columns={county_col: "county", rate_col: "obesity_rate"})
    df = df.merge(ob[["county", "obesity_rate"]], on="county", how="left")
else:
    df["obesity_rate"] = np.nan

# Placeholder metrics (until you add ACS-by-ZIP)
# These keep the app working + demoable while we plug real sources.
rng = np.random.default_rng(42)
df["median_income"] = rng.normal(85000, 12000, len(df)).clip(35000, 160000)
df["insurance_coverage"] = rng.normal(0.92, 0.03, len(df)).clip(0.75, 0.99) * 100
df["population_density"] = rng.normal(1800, 900, len(df)).clip(50, 9000)

# Obesity: if we have real values, use them; else synthesize
if df["obesity_rate"].isna().all():
    df["obesity_rate"] = rng.normal(28, 4, len(df)).clip(18, 40)

# -----------------------
# Scoring per business model
# -----------------------
# If models.json is present, we use weights from there; else use defaults.
default_weights = {
    "Urgent Care (Premium)": {"median_income": 0.35, "insurance_coverage": 0.40, "population_density": 0.20, "obesity_rate": -0.05},
    "Gym (Premium)":         {"median_income": 0.45, "insurance_coverage": 0.15, "population_density": 0.30, "obesity_rate": -0.10},
    "Gym (Budget)":          {"median_income": -0.10, "insurance_coverage": 0.05, "population_density": 0.55, "obesity_rate": 0.50},
    "Fast Food (Budget)":    {"median_income": -0.35, "insurance_coverage": -0.05, "population_density": 0.40, "obesity_rate": 0.60},
    "Fine Dining":           {"median_income": 0.60, "insurance_coverage": 0.15, "population_density": 0.25, "obesity_rate": -0.10},
}

weights = None
if isinstance(models, dict) and chosen_model in models and isinstance(models[chosen_model], dict):
    weights = models[chosen_model].get("weights")
if not weights:
    weights = default_weights.get(chosen_model, default_weights["Urgent Care (Premium)"])

# Normalize metrics and compute score
score = 0
for feat, w in weights.items():
    if feat not in df.columns:
        continue
    val = normalize_0_100(df[feat])
    score += w * val

df["score"] = normalize_0_100(pd.Series(score))

# Optional "animation" = reveal by deciles
if animate:
    df["band"] = pd.qcut(df["score"], 10, labels=[f"D{i}" for i in range(1, 11)])
    df["band_order"] = df["band"].str.replace("D", "").astype(int)
else:
    df["band"] = "All"
    df["band_order"] = 1

# -----------------------
# Map
# -----------------------
st.subheader("ZIP Heatmap")

# Plotly needs a featureid key path to match zcta values
featureid_key = f"properties.{zip_field}"

fig = px.choropleth_mapbox(
    df.sort_values("band_order"),
    geojson=geo,
    locations="zcta",
    featureidkey=featureid_key,
    color="score",
    hover_data={
        "zcta": True,
        "score": ':.1f',
        "median_income": ':.0f',
        "insurance_coverage": ':.1f',
        "population_density": ':.0f',
        "obesity_rate": ':.1f',
        "band": True
    },
    animation_frame="band" if animate else None,
    mapbox_style=map_style,
    zoom=7.2,
    center={"lat": 41.6, "lon": -72.7},
    opacity=0.75,
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

st.caption("Nota: enquanto não plugamos ACS por ZIP, alguns indicadores estão como placeholder só para validar o produto.")
if not has_crosswalk:
    st.info("Para aplicar obesidade (county) em ZIP, adicione um arquivo zcta_to_county.csv com colunas: zcta, county.")
