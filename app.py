import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="CT ZIP Heatmap Platform", layout="wide")

# ---------- Files ----------
GEOJSON_PATH = "ct_zcta.json"
ACS_PATH = "acs_by_zcta.csv"
XWALK_PATH = "zcta_to_county.csv"
OBESITY_PATH = "obesity_by_county.csv"
MODELS_PATH = "models.json"

# ---------- Helpers ----------
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_zip_field(geo: dict) -> str:
    props = geo["features"][0].get("properties", {})
    for k in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"]:
        if k in props:
            return k
    for k in props.keys():
        if "ZCTA" in k.upper():
            return k
    return ""

def normalize_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series([50] * len(s), index=s.index)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if mn == mx:
        return pd.Series([50] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# ---------- Load core geo ----------
if not os.path.exists(GEOJSON_PATH):
    st.error(f"Faltando {GEOJSON_PATH} no repo (na raiz).")
    st.stop()

geo = load_geojson(GEOJSON_PATH)
zip_field = detect_zip_field(geo)
if not zip_field:
    st.error("Não consegui detectar o campo do ZIP no GeoJSON (esperava algo tipo ZCTA5CE20).")
    st.stop()

# Base ZIP list from polygons
zips = []
for feat in geo["features"]:
    z = feat.get("properties", {}).get(zip_field, "")
    if z:
        zips.append(str(z).zfill(5))
base = pd.DataFrame({"zcta": sorted(set(zips))})

# ---------- Load data tables ----------
acs = safe_read_csv(ACS_PATH)
xwalk = safe_read_csv(XWALK_PATH)
ob = safe_read_csv(OBESITY_PATH)

# Normalize expected columns if present
def normalize_cols(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    for want, candidates in mapping.items():
        for cand in candidates:
            if cand.lower() in cols and cols[cand.lower()] != want:
                df = df.rename(columns={cols[cand.lower()]: want})
                break
    return df

acs = normalize_cols(acs, {
    "zcta": ["zcta", "zip", "zipcode"],
    "median_income": ["median_income", "income", "median_household_income"],
    "median_age": ["median_age", "age"],
    "population": ["population", "pop"]
})

xwalk = normalize_cols(xwalk, {
    "zcta": ["zcta", "zip", "zipcode"],
    "county_fips": ["county_fips", "fips", "countyfp", "geoid"],
    "county_name": ["county_name", "county", "name"]
})

ob = normalize_cols(ob, {
    "county_fips": ["county_fips", "fips", "countyfp", "geoid"],
    "county_name": ["county_name", "county", "name"],
    "obesity_rate": ["obesity_rate", "obesity", "rate", "percent", "pct", "value"]
})

# Force formats
for df in [acs, xwalk]:
    if not df.empty and "zcta" in df.columns:
        df["zcta"] = df["zcta"].astype(str).str.zfill(5)

# ---------- Merge into one ZIP table ----------
df = base.copy()

if not acs.empty and "zcta" in acs.columns:
    df = df.merge(acs[["zcta"] + [c for c in ["median_income","median_age","population"] if c in acs.columns]],
                  on="zcta", how="left")

if not xwalk.empty and "zcta" in xwalk.columns:
    keep_cols = ["zcta"] + [c for c in ["county_fips","county_name"] if c in xwalk.columns]
    df = df.merge(xwalk[keep_cols], on="zcta", how="left")

# Join obesity via county_fips (preferred) else county_name
if not ob.empty:
    if "county_fips" in df.columns and "county_fips" in ob.columns:
        df = df.merge(ob[["county_fips","obesity_rate"]].drop_duplicates(), on="county_fips", how="left")
    elif "county_name" in df.columns and "county_name" in ob.columns:
        df = df.merge(ob[["county_name","obesity_rate"]].drop_duplicates(), on="county_name", how="left")

# ---------- If missing real data, fill placeholders (so app still works) ----------
rng = np.random.default_rng(42)

if "median_income" not in df.columns:
    df["median_income"] = np.nan
if "median_age" not in df.columns:
    df["median_age"] = np.nan
if "population" not in df.columns:
    df["population"] = np.nan
if "obesity_rate" not in df.columns:
    df["obesity_rate"] = np.nan

missing_real = df[["median_income","median_age","population"]].isna().all().all()

if missing_real:
    # placeholders (demo mode)
    df["median_income"] = rng.normal(85000, 12000, len(df)).clip(35000, 160000)
    df["median_age"] = rng.normal(40, 6, len(df)).clip(22, 65)
    df["population"] = rng.normal(16000, 9000, len(df)).clip(500, 80000)

if df["obesity_rate"].isna().all():
    df["obesity_rate"] = rng.normal(28, 4, len(df)).clip(18, 42)

# Derived: insurance coverage placeholder for now (até ACS insurance entrar)
df["insurance_coverage"] = rng.normal(0.92, 0.03, len(df)).clip(0.75, 0.99) * 100

# ---------- Models ----------
models = {}
if os.path.exists(MODELS_PATH):
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            models = json.load(f)
    except Exception:
        models = {}

default_weights = {
    "Urgent Care (Premium)": {"median_income": 0.35, "insurance_coverage": 0.40, "population": 0.25, "obesity_rate": -0.05},
    "Gym (Premium)":         {"median_income": 0.45, "insurance_coverage": 0.10, "population": 0.35, "obesity_rate": -0.10},
    "Gym (Budget)":          {"median_income": -0.10, "insurance_coverage": 0.05, "population": 0.45, "obesity_rate": 0.60},
    "Fast Food (Budget)":    {"median_income": -0.35, "insurance_coverage": -0.05, "population": 0.30, "obesity_rate": 0.75},
    "Fine Dining":           {"median_income": 0.65, "insurance_coverage": 0.10, "population": 0.25, "obesity_rate": -0.10},
}

st.title("CT ZIP Heatmap Platform")

left, mid, right = st.columns([1.2, 1, 1])
with left:
    model_names = list(models.keys()) if isinstance(models, dict) and models else list(default_weights.keys())
    chosen_model = st.selectbox("Modelo de negócio", model_names, index=0)
with mid:
    animate = st.toggle("Animação (reveal por faixas)", value=True)
with right:
    map_style = st.selectbox("Mapa", ["carto-positron", "open-street-map"], index=0)

# Choose weights
weights = None
if isinstance(models, dict) and chosen_model in models and isinstance(models[chosen_model], dict):
    weights = models[chosen_model].get("weights")
if not weights:
    weights = default_weights.get(chosen_model, default_weights["Urgent Care (Premium)"])

# Score
score = 0
for feat, w in weights.items():
    if feat not in df.columns:
        continue
    score += w * normalize_0_100(df[feat])
df["score"] = normalize_0_100(pd.Series(score))

# Animation bands
if animate:
    df["band"] = pd.qcut(df["score"], 10, labels=[f"D{i}" for i in range(1, 11)])
    df["band_order"] = df["band"].astype(str).str.replace("D","").astype(int)
else:
    df["band"] = "All"
    df["band_order"] = 1

# ---------- Map ----------
st.subheader("Heatmap por ZIP (polígono ZCTA)")
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
        "median_age": ':.1f',
        "population": ':.0f',
        "insurance_coverage": ':.1f',
        "obesity_rate": ':.1f',
        "band": True,
    },
    animation_frame="band" if animate else None,
    mapbox_style=map_style,
    zoom=7.2,
    center={"lat": 41.6, "lon": -72.7},
    opacity=0.78,
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

if missing_real:
    st.info("Modo demo: ACS por ZIP ainda não foi preenchido (acs_by_zcta.csv está só com header). Quando você subir o CSV real, o mapa vira 100% dados reais automaticamente.")
if xwalk.empty or ("county_fips" not in df.columns and "county_name" not in df.columns):
    st.warning("zcta_to_county.csv está vazio (só header). Obesidade por county vai entrar automaticamente quando esse arquivo tiver dados.")
