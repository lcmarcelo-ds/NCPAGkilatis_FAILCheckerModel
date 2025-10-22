import re
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import os

# ----- PyDeck/Streamlit stability fix -----
os.environ["PYDECK_USE_LEGACY_API"] = "True"

st.set_page_config(
    page_title="FAIL Checker Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Configuration ----------
DATA_PATH_DEFAULT = "FAIL_Checker_Model_Output.xlsx"

CATEGORIES = [
    "Green_Flag",
    "Siyam-siyam_Project",
    "Chop_chop_Project",
    "Doppelganger_Project",
    "Ghost_Project",
]
GREEN_SET = {"Green_Flag"}
RED_SET = {"Siyam-siyam_Project", "Chop_chop_Project", "Doppelganger_Project", "Ghost_Project"}

CAT_COLOR: Dict[str, List[int]] = {
    "Green_Flag": [18, 152, 66, 190],
    "Siyam-siyam_Project": [214, 139, 0, 190],
    "Chop_chop_Project":   [189, 28, 28, 190],
    "Doppelganger_Project":[128, 60, 170, 190],
    "Ghost_Project":       [32, 96, 168, 190],
}
MAP_INITIAL_VIEW = pdk.ViewState(latitude=12.8797, longitude=121.7740, zoom=5, pitch=0)

# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names safely."""
    new_cols = {c: re.sub(r"\s+", "_", str(c).strip()) for c in df.columns}
    return df.rename(columns=new_cols)

def find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    # exact match
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # fuzzy contains
    for c in cols:
        lc = c.lower()
        if any(k.lower() in lc for k in candidates):
            return c
    return None

# column detectors
def detect_project_id_col(df):  return find_column(df, ["ProjectID", "Project_Id", "ProjID", "ProjectCode"])
def detect_lat_lon_cols(df):    return find_column(df, ["Latitude","Lat"]), find_column(df, ["Longitude","Lon","Lng"])
def detect_deo_col(df):         return find_column(df, ["DistrictEngineeringOffice","District_Engineering_Office","DEO"])
def detect_contractor_col(df):  return find_column(df, ["Contractor","ContractorName","Supplier"])
def detect_year_col(df):        return find_column(df, ["InfraYear","StartYear","CompletionYear"])
def detect_cost_col(df):
    col = find_column(df, ["ContractCost","ProjectCost","Contract_Amount","Total_Contract_Cost"])
    if col: return col
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None
def detect_tag_col(df):
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(400).str.lower()
        if vals.empty:
            continue
        if any(any(cat.lower() in v for v in vals) for cat in CATEGORIES):
            return c
    return find_column(df, ["Tags","Tagging","Category","Labels"])

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def deduplicate_by_last(df: pd.DataFrame, key_col: Optional[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=[key_col], keep="last") if key_col and key_col in df.columns else df

def explode_tags(df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    t = df.copy()
    t[tag_col] = t[tag_col].astype(str)
    t["_tags_list"] = t[tag_col].str.split(r"\s*;\s*")
    t = t.explode("_tags_list")
    t["_tags_list"] = t["_tags_list"].str.strip()
    t = t[t["_tags_list"].isin(CATEGORIES)]
    return t.rename(columns={"_tags_list": "Tag"})

def contains_category(tag_string: str, category: str) -> bool:
    return bool(re.search(fr"(^|;)\s*{re.escape(category)}\s*(;|$)", str(tag_string), flags=re.IGNORECASE))

def filter_inclusive(df: pd.DataFrame, tag_col: str, category: str) -> pd.DataFrame:
    return df[df[tag_col].astype(str).apply(lambda s: contains_category(s, category))].copy()

def add_year_if_missing(df: pd.DataFrame, year_col: Optional[str]):
    if year_col:
        return df, year_col
    for c in df.columns:
        if "date" in c.lower() or "year" in c.lower():
            try:
                d = pd.to_datetime(df[c], errors="coerce")
                if d.notna().sum() > 0:
                    df2 = df.copy()
                    df2["__Year"] = d.dt.year
                    return df2, "__Year"
            except Exception:
                continue
    return df, None

def alt_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, sort_col: Optional[str] = None, topn: Optional[int] = None):
    d = df.copy()
    if sort_col:
        d = d.sort_values(sort_col, ascending=False)
    if topn:
        d = d.head(topn)
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(y_col, title=y_col, type="quantitative"),
            y=alt.Y(x_col, sort="-x", title=x_col),
            tooltip=[x_col, y_col],
        )
        .properties(height=400, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

def top_entities(df: pd.DataFrame, deo_col: Optional[str], contractor_col: Optional[str], cost_col: Optional[str], by="count", topn=15):
    out = {}
    if deo_col and deo_col in df.columns:
        if by == "cost" and cost_col:
            tmp = df.copy()
            tmp["_cost"] = ensure_numeric(tmp[cost_col])
            s = tmp.groupby(deo_col, dropna=False)["_cost"].sum().sort_values(ascending=False).head(topn)
            out["DEO"] = s.reset_index(name="Total Cost")
        else:
            s = df.groupby(deo_col, dropna=False).size().sort_values(ascending=False).head(topn)
            out["DEO"] = s.reset_index(name="Projects")
    if contractor_col and contractor_col in df.columns:
        if by == "cost" and cost_col:
            tmp = df.copy()
            tmp["_cost"] = ensure_numeric(tmp[cost_col])
            s = tmp.groupby(contractor_col, dropna=False)["_cost"].sum().sort_values(ascending=False).head(topn)
            out["Contractor"] = s.reset_index(name="Total Cost")
        else:
            s = df.groupby(contractor_col, dropna=False).size().sort_values(ascending=False).head(topn)
            out["Contractor"] = s.reset_index(name="Projects")
    return out

def map_layers_for_categories(df, lat_col, lon_col, tag_col, cats):
    layers = []
    for cat in cats:
        dd = df[df[tag_col].astype(str).apply(lambda s: contains_category(s, cat))]
        dd = dd.dropna(subset=[lat_col, lon_col])
        if dd.empty:
            continue
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=dd,
                get_position=[lon_col, lat_col],
                get_radius=80,
                pickable=True,
                radius_min_pixels=3,
                radius_max_pixels=8,
                get_fill_color=CAT_COLOR.get(cat, [120,120,120,160]),
                auto_highlight=True,
            )
        )
    return layers

def deck_chart(layers, tooltip=None):
    """Return a safe Deck object; avoid SessionInfo serialization crashes."""
    try:
        return pdk.Deck(initial_view_state=MAP_INITIAL_VIEW, layers=layers, tooltip=tooltip)
    except Exception as e:
        st.warning(f"Map rendering temporarily failed: {e}")
        # Return an empty but valid deck object so the app continues to run
        return pdk.Deck(initial_view_state=MAP_INITIAL_VIEW, layers=[], tooltip=tooltip)

# ---------- Load data ----------
try:
    df_raw = load_data(DATA_PATH_DEFAULT)
except Exception as e:
    st.error(f"Failed to read Excel file '{DATA_PATH_DEFAULT}'. Error: {e}")
    st.stop()

df = normalize_columns(df_raw)
proj_col = detect_project_id_col(df)
lat_col, lon_col = detect_lat_lon_cols(df)
deo_col = detect_deo_col(df)
contractor_col = detect_contractor_col(df)
cost_col = detect_cost_col(df)
tag_col = detect_tag_col(df)
year_col = detect_year_col(df)

missing = []
if tag_col is None: missing.append("Tag/Category column (Green_Flag, etc.)")
if lat_col is None or lon_col is None: missing.append("Latitude/Longitude")
if deo_col is None: missing.append("DistrictEngineeringOffice")
if missing:
    st.error("Missing required columns: " + "; ".join(missing))
    st.stop()

df = deduplicate_by_last(df, proj_col)
df, derived_year_col = add_year_if_missing(df, year_col)
year_col = year_col or derived_year_col
working = df.copy()

# ---------- UI Tabs ----------
tabs = st.tabs(["Overview", "Compare", "Green Flag", "Siyam-siyam", "Chop-chop", "Doppelganger", "Ghost"])

# ===== Overview =====
with tabs[0]:
    st.header("Overview")
    total_projects = len(working)
    exploded = explode_tags(working, tag_col)
    tag_counts = exploded["Tag"].value_counts().reindex(CATEGORIES, fill_value=0)
    green_count = int(tag_counts.get("Green_Flag", 0))
    red_count = int(tag_counts.reindex(list(RED_SET), fill_value=0).sum())
    green_share = (green_count / total_projects * 100) if total_projects > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", f"{total_projects:,}")
    c2.metric("Green Flag Projects", f"{green_count:,}")
    c3.metric("Red-tagged Projects", f"{red_count:,}")
    c4.metric("Green Share (%)", f"{green_share:.2f}")

    dist_df = tag_counts.rename_axis("Tag").reset_index(name="Projects")
    alt_bar(dist_df, x_col="Tag", y_col="Projects", title="Distribution of Tags", sort_col="Projects")

    st.subheader("Comparison Map — Green vs Red")
    d = working.copy()
    d["_is_green"] = d[tag_col].astype(str).apply(lambda s: contains_category(s, "Green_Flag"))
    d["_is_red"] = d[tag_col].astype(str).apply(lambda s: any(contains_category(s, c) for c in RED_SET))
    greens = d[d["_is_green"] & d[lat_col].notna() & d[lon_col].notna()]
    reds = d[d["_is_red"] & d[lat_col].notna() & d[lon_col].notna()]
    deck = deck_chart(
        layers=[
            pdk.Layer("ScatterplotLayer", data=greens, get_position=[lon_col, lat_col],
                      get_radius=80, pickable=True, get_fill_color=CAT_COLOR["Green_Flag"]),
            pdk.Layer("ScatterplotLayer", data=reds, get_position=[lon_col, lat_col],
                      get_radius=80, pickable=True, get_fill_color=[200, 40, 40, 190]),
        ],
        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
    )
    st.pydeck_chart(deck)

    st.subheader("Top Entities")
    mode = st.radio("Rank by", ["Project Count", "Total Cost"], horizontal=True, index=0)
    mode_key = "count" if mode == "Project Count" else "cost"
    tops = top_entities(working, deo_col, contractor_col, cost_col, by=mode_key, topn=15)
    t1, t2 = st.columns(2)
    if "DEO" in tops: t1.dataframe(tops["DEO"])
    if "Contractor" in tops: t2.dataframe(tops["Contractor"])

# ===== Compare =====
with tabs[1]:
    st.header("Category Comparison Map")
    layers = map_layers_for_categories(working, lat_col, lon_col, tag_col, CATEGORIES)
    st.pydeck_chart(deck_chart(layers))

# ===== Category Tabs (Green, Siyam, Chop, Doppelganger, Ghost) =====
def render_category_tab(df_in, category, container):
    with container:
        st.header(category.replace("_"," "))
        filtered = filter_inclusive(df_in, tag_col, category)
        count = len(filtered)
        st.metric(f"{category} Projects", f"{count:,}")

        st.dataframe(filtered)  # show details
        layers = map_layers_for_categories(df_in, lat_col, lon_col, tag_col, [category])
        st.pydeck_chart(deck_chart(layers))

render_category_tab(working, "Green_Flag", tabs[2])
render_category_tab(working, "Siyam-siyam_Project", tabs[3])
render_category_tab(working, "Chop_chop_Project", tabs[4])
render_category_tab(working, "Doppelganger_Project", tabs[5])
render_category_tab(working, "Ghost_Project", tabs[6])
