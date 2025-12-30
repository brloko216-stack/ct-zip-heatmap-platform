import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CT ZIP Heatmap Platform", layout="wide")

# ----------------------------
# Login simples
# ----------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "demo123")
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title("Client Login")
    st.caption("Client-restricted access – Connecticut only")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == APP_PASSWORD:
            st.session_state.authed = True
        else:
            st.error("Wrong password")
    st.markdown("""
    **Notes**
    - Indicative scores
    - Decision-support only
    - Aggregated public data (CT)
    """)
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
REQUIRED_FEATURES = [
    "zcta", "county",
    "median_income",          # ACS (por ZIP/ZCTA)
    "insurance_coverage",     # ACS (por ZIP/ZCTA) ou proxy
    "population_density",     # proxy por ZIP (pop/area) ou outro
    "age_18_34",
    "age_25_64",
    "age_35_64",
    "low_income_index",       # 0-100 (você define)
    "accessibility",          # 0-100 (proxy: drive time / highways etc.)
    "competition"             # 0-100 (maior = mais concorrência)
]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([50] * len(series), index=series.index)  # fallback neutro
    s = s.clip(lower=lo, upper=hi)
    return ((s - lo) / (hi - lo) * 100).round(2)

def score_badge(score: int) -> str:
    if score >= 82:
        return "PROCEED"
    if score >= 72:
        return "CAUTION"
    return "AVOID"

def safe_read_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def find_featureidkey(geojson: dict, candidate_keys: list[str]) -> str | None:
    # tenta achar qual propriedade existe no primeiro feature
    try:
        props = geojson["features"][0]["properties"]
    except Exception:
        return None
    for k in candidate_keys:
        if k in props:
            return f"properties.{k}"
    return None

# ----------------------------
# Load inputs
# ----------------------------
if not os.path.exists("models.json"):
    st.error("Missing models.json in repo.")
    st.stop()

MODELS = load_json("models.json")

zcta_df = safe_read_csv("ct_zcta_features.csv")  # << seu dataset REAL por ZIP/ZCTA
ob_df = safe_read_csv("obesity_by_county.csv")   # << seu dataset REAL por county

# Sidebar controls
st.sidebar.title("CT Heatmap Platform")
model_key = st.sidebar.selectbox(
    "Tipo de negócio",
    list(MODELS.keys()),
    format_func=lambda k: MODELS[k].get("label", k),
)

geo_mode = st.sidebar.radio("Mapa por", ["ZIP (ZCTA)", "County"], index=0)

st.sidebar.divider()
st.sidebar.caption("Cores = pontuação (0–100).")

# ----------------------------
# Validate dataset
# ----------------------------
if zcta_df is None:
    st.error("Faltou o arquivo **ct_zcta_features.csv** no repo. Adicione ele e tente novamente.")
    st.stop()

missing_cols = [c for c in REQUIRED_FEATURES if c not in zcta_df.columns]
if missing_cols:
    st.error(f"ct_zcta_features.csv está faltando colunas: {missing_cols}")
    st.stop()

df = zcta_df.copy()
df["zcta"] = df["zcta"].astype(str).str.zfill(5)

# Merge obesity by county (optional)
if ob_df is not None and "county" in ob_df.columns and "obesity_rate" in ob_df.columns:
    df = df.merge(ob_df[["county", "obesity_rate"]], on="county", how="left")
else:
    df["obesity_rate"] = pd.NA  # não quebra o app

# Derived: competition inverted (melhor quando concorrência é menor)
df["competition_inverted"] = 100 - pd.to_numeric(df["competition"], errors="coerce")

# ----------------------------
# Normalize features to 0-100
# (para que pesos funcionem consistentemente)
# ----------------------------
# Aqui você pode adicionar/remover features conforme seu dataset REAL.
norm_cols = [
    "median_income",
    "insurance_coverage",
    "population_density",
    "age_18_34",
    "age_25_64",
    "age_35_64",
    "low_income_index",
    "accessibility",
    "obesity_rate",
    "competition_inverted"
]

for col in norm_cols:
    if col not in df.columns:
        df[col] = pd.NA
    df[col + "_n"] = normalize_0_100(df[col])

# ----------------------------
# Score engine (model weights)
# ----------------------------
weights = MODELS[model_key]["weights"]

# Weights referem-se aos nomes "base" (sem _n). Ex: median_income -> median_income_n
needed = []
for feat in weights.keys():
    if feat.endswith("_inverted"):
        base = feat  # competition_inverted
    else:
        base = feat
    if base + "_n" not in df.columns:
        needed.append(base)

if needed:
    st.error(f"Modelo exige features que não existem no dataset: {needed}")
    st.stop()

df["final_score"] = 0.0
for feat, w in weights.items():
    df["final_score"] += df[feat + "_n"] * float(w)

df["final_score"] = df["final_score"].round(0).astype(int)
df["recommendation"] = df["final_score"].apply(score_badge)

# ----------------------------
# Header
# ----------------------------
st.title("Connecticut Heatmap (ZIP / County)")
st.caption(f"Modelo: **{MODELS[model_key].get('label', model_key)}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg score", int(df["final_score"].mean()))
c2.metric("Top score", int(df["final_score"].max()))
c3.metric("Bottom score", int(df["final_score"].min()))
c4.metric("ZIPs", df["zcta"].nunique())

st.divider()

# ----------------------------
# Table views
# ----------------------------
top = df.sort_values("final_score", ascending=False).head(15)
bottom = df.sort_values("final_score", ascending=True).head(15)

t1, t2 = st.tabs(["Top ZIPs", "Bottom ZIPs"])
with t1:
    st.dataframe(top[["zcta", "county", "final_score", "recommendation"]], use_container_width=True)
with t2:
    st.dataframe(bottom[["zcta", "county", "final_score", "recommendation"]], use_container_width=True)

st.divider()

# ----------------------------
# Map (ZIP choropleth preferred)
# ----------------------------
center_ct = {"lat": 41.6, "lon": -72.7}

if geo_mode == "ZIP (ZCTA)":
    if not os.path.exists("ct_zcta_ct.geojson"):
        st.warning("Faltou **ct_zcta_ct.geojson** (polígonos ZCTA de CT). Sem ele não dá pra pintar ZIP por cor.")
        st.stop()

    with open("ct_zcta_ct.geojson", "r", encoding="utf-8") as f:
        geo = json.load(f)

    # Chaves comuns dentro de GeoJSON de ZCTA
    featureidkey = find_featureidkey(geo, ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE00", "ZCTA5"])
    if featureidkey is None:
        st.error("Não consegui achar a chave do ZCTA dentro do GeoJSON. Me mande um print do primeiro feature/properties.")
        st.stop()

    fig = px.choropleth_mapbox(
        df,
        geojson=geo,
        locations="zcta",
        featureidkey=featureidkey,
        color="final_score",
        hover_data={"county": True, "recommendation": True, "final_score": True},
        mapbox_style="open-street-map",
        zoom=7.2,
        center=center_ct,
        opacity=0.70,
        height=620
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

else:
    # County choropleth
    if not os.path.exists("ct_county_ct.geojson"):
        st.warning("Faltou **ct_county_ct.geojson** (polígonos de county). Adicione e ele pinta por county.")
        st.stop()

    with open("ct_county_ct.geojson", "r", encoding="utf-8") as f:
        cgeo = json.load(f)

    # Agrega score por county (média)
    cdf = df.groupby("county", as_index=False).agg(
        final_score=("final_score", "mean"),
        zips=("zcta", "nunique")
    )
    cdf["final_score"] = cdf["final_score"].round(0).astype(int)
    cdf["recommendation"] = cdf["final_score"].apply(score_badge)

    featureidkey = find_featureidkey(cgeo, ["NAME", "NAMELSAD", "COUNTYNAME"])
    if featureidkey is None:
        st.error("Não achei o nome do county dentro do GeoJSON (properties). Me mande um print do primeiro feature/properties.")
        st.stop()

    fig = px.choropleth_mapbox(
        cdf,
        geojson=cgeo,
        locations="county",
        featureidkey=featureidkey,
        color="final_score",
        hover_data={"zips": True, "recommendation": True, "final_score": True},
        mapbox_style="open-street-map",
        zoom=7.2,
        center=center_ct,
        opacity=0.70,
        height=620
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Export
# ----------------------------
st.download_button(
    "Download scored ZIPs (CSV)",
    data=df[["zcta", "county", "final_score", "recommendation"]].to_csv(index=False).encode("utf-8"),
    file_name="ct_scored_zips.csv",
    mime="text/csv",
)
