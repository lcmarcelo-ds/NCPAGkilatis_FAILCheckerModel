# Streamlit Dashboard for FAIL Checker Model Output
# Author: ChatGPT
# Description: Comprehensive dashboard with Overall section and per-category analyses,
# including maps, distributions, top entities, and per-DEO metrics.

import re
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -------------------------------
# Config
# -------------------------------
st.set_page_config(
    page_title="FAIL Checker Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH_DEFAULT = "FAIL_Checker_Model_Output.xlsx"

CATEGORIES = [
    "Green_Flag",
    "Siyam-siyam_Project",
    "Chop_chop_Project",
    "Dopplelganger_Project",
    "Ghost_Project",
]
GREEN_SET = {"Green_Flag"}
RED_SET = {"Siyam-siyam_Project", "Chop_chop_Project", "Dopplelganger_Project", "Ghost_Project"}

MAP_INITIAL_VIEW = pdk.ViewState(latitude=12.8797, longitude=121.7740, zoom=5, pitch=0)

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Read Excel (first sheet by default)
    df = pd.read_excel(path)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip spaces and unify column names to snake_case-like (no mutation of original)
    new_cols = {}
    for c in df.columns:
        # replace whitespace with underscore, drop leading/trailing underscores
        nc = re.sub(r"\s+", "_", str(c)).strip("_")
        new_cols[c] = nc
    return df.rename(columns=new_cols)

def find_column(df: pd.DataFrame, candidates, must_contain_any=None):
    cols = list(df.columns)
    # exact
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # fuzzy contains
    for c in cols:
        lc = c.lower()
        if any(k.lower() in lc for k in candidates):
            if not must_contain_any or any(k.lower() in lc for k in must_contain_any):
                return c
    return None

def detect_project_id_col(df: pd.DataFrame):
    return find_column(df, ["ProjectID", "Project_Id", "Project_ID", "ProjID", "ProjectCode", "Project_Code"])

def detect_lat_lon_cols(df: pd.DataFrame):
    lat_col = find_column(df, ["Latitude", "Lat"])
    lon_col = find_column(df, ["Longitude", "Lon", "Lng"])
    return lat_col, lon_col

def detect_deo_col(df: pd.DataFrame):
    return find_column(df, ["DistrictEngineeringOffice", "District_Engineering_Office", "DEO"])

def detect_contractor_col(df: pd.DataFrame):
    return find_column(df, ["Contractor", "ContractorName", "Supplier", "SupplierName"])

def detect_cost_col(df: pd.DataFrame):
    # Prefer common names first
    candidates = [
        "ContractCost",
        "Total_Contract_Cost",
        "Contract_Amount",
        "ProjectCost",
        "Approved_Budget_for_the_Contract",
        "ABC",
    ]
    col = find_column(df, candidates)
    if col:
        return col
    # Fallback: keyword scan
    for c in df.columns:
        lc = c.lower()
        if ("contract" in lc or "project" in lc) and ("cost" in lc or "amount" in lc or "value" in lc):
            return c
    # Last resort: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def detect_tag_col(df: pd.DataFrame):
    # Find a column that clearly contains our category labels
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(200).str.lower()
        if vals.empty:
            continue
        if any(any(cat.lower() in v for v in vals) for cat in CATEGORIES):
            return c
    # Fallback: name-based
    return find_column(df, ["Tags", "Tagging", "FAIL_Tags", "FAIL_Tag", "Category", "Categories", "Labels"])

def ensure_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0)

def deduplicate_by_last(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if key_col and key_col in df.columns:
        return df.drop_duplicates(subset=[key_col], keep="last")
    return df

def explode_tags(df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    temp = df.copy()
    temp[tag_col] = temp[tag_col].astype(str)
    temp["_tags_list"] = temp[tag_col].str.split(r"\s*;\s*")
    temp = temp.explode("_tags_list")
    temp["_tags_list"] = temp["_tags_list"].str.strip()
    temp = temp[temp["_tags_list"].isin(CATEGORIES)]
    return temp.rename(columns={"_tags_list": "Tag"})

def filter_by_category_inclusive(df: pd.DataFrame, tag_col: str, category: str) -> pd.DataFrame:
    # Include rows where the semicolon-delimited tag string contains `category`
    mask = df[tag_col].astype(str).str.contains(fr"(^|;)\s*{re.escape(category)}\s*(;|$)", case=False, na=False)
    return df[mask].copy()

def compute_green_metrics(df: pd.DataFrame, deo_col: str, cost_col: str, tag_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_is_green"] = work[tag_col].astype(str).str.contains(r"(^|;)\s*Green_Flag\s*(;|$)", case=False, na=False)
    work["_cost"] = ensure_numeric(work[cost_col]) if cost_col else 0

    # Totals per DEO
    tot = work.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    # Green counts
    green_cnt = work.groupby(deo_col, dropna=False)["_is_green"].sum().astype(int).reset_index(name="green_flag_projects")
    # Costs
    if cost_col:
        costs = work.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        green_costs = (
            work[work["_is_green"]]
            .groupby(deo_col, dropna=False)["_cost"]
            .sum()
            .reset_index(name="green_flag_contract_cost")
        )
    else:
        costs = pd.DataFrame({deo_col: work[deo_col].unique(), "total_contract_cost": 0})
        green_costs = pd.DataFrame({deo_col: work[deo_col].unique(), "green_flag_contract_cost": 0})

    out = (
        tot.merge(green_cnt, on=deo_col, how="left")
           .merge(costs, on=deo_col, how="left")
           .merge(green_costs, on=deo_col, how="left")
           .fillna({"green_flag_projects": 0, "green_flag_contract_cost": 0, "total_contract_cost": 0})
    )
    out["green_flag_projects"] = out["green_flag_projects"].astype(int)
    out["green_flag_density"] = np.where(out["total_projects"] > 0,
                                         out["green_flag_projects"] / out["total_projects"], 0.0)
    out["green_flag_cost_ratio"] = np.where(out["total_contract_cost"] > 0,
                                            out["green_flag_contract_cost"] / out["total_contract_cost"], 0.0)
    out = out.sort_values(["green_flag_projects", "green_flag_density"], ascending=[False, False]).reset_index(drop=True)
    return out

def compute_category_metrics(df: pd.DataFrame, deo_col: str, cost_col: str, tag_col: str, category: str) -> pd.DataFrame:
    """
    Generic per-DEO metrics for any category (e.g., Siyam-siyam, Chop-chop, Dopplelganger, Ghost).
    Outputs: total_projects, category_projects, total_contract_cost, category_contract_cost,
             category_density, category_cost_ratio
    """
    work = df.copy()
    pattern = fr"(^|;)\s*{re.escape(category)}\s*(;|$)"
    work["_in_cat"] = work[tag_col].astype(str).str.contains(pattern, case=False, na=False)
    work["_cost"] = ensure_numeric(work[cost_col]) if cost_col else 0

    tot = work.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    cat_cnt = work.groupby(deo_col, dropna=False)["_in_cat"].sum().astype(int).reset_index(name="category_projects")

    if cost_col:
        costs = work.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        cat_costs = (
            work[work["_in_cat"]]
            .groupby(deo_col, dropna=False)["_cost"]
            .sum()
            .reset_index(name=f"{category}_contract_cost")
        )
    else:
        costs = pd.DataFrame({deo_col: work[deo_col].unique(), "total_contract_cost": 0})
        cat_costs = pd.DataFrame({deo_col: work[deo_col].unique(), f"{category}_contract_cost": 0})

    out = (
        tot.merge(cat_cnt, on=deo_col, how="left")
           .merge(costs, on=deo_col, how="left")
           .merge(cat_costs, on=deo_col, how="left")
           .fillna({"category_projects": 0, f"{category}_contract_cost": 0, "total_contract_cost": 0})
    )
    out["category_projects"] = out["category_projects"].astype(int)
    out["category_density"] = np.where(out["total_projects"] > 0,
                                       out["category_projects"] / out["total_projects"], 0.0)
    out["category_cost_ratio"] = np.where(out["total_contract_cost"] > 0,
                                          out[f"{category}_contract_cost"] / out["total_contract_cost"], 0.0)
    out = out.sort_values(["category_projects", "category_density"], ascending=[False, False]).reset_index(drop=True)
    return out

def top_entities(df: pd.DataFrame, deo_col: str, contractor_col: str, cost_col: str | None, by="count", topn=15):
    out = {}
    if deo_col and deo_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = (df.groupby(deo_col, dropna=False)[cost_col]
                     .apply(ensure_numeric).sum()
                     .sort_values(ascending=False).head(topn).reset_index(name="Total Cost"))
        else:
            tmp = (df.groupby(deo_col, dropna=False)
                     .size().sort_values(ascending=False).head(topn).reset_index(name="Projects"))
        out["DistrictEngineeringOffice"] = tmp
    if contractor_col and contractor_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = (df.groupby(contractor_col, dropna=False)[cost_col]
                     .apply(ensure_numeric).sum()
                     .sort_values(ascending=False).head(topn).reset_index(name="Total Cost"))
        else:
            tmp = (df.groupby(contractor_col, dropna=False)
                     .size().sort_values(ascending=False).head(topn).reset_index(name="Projects"))
        out["Contractor"] = tmp
    return out

def make_green_red_layers(df: pd.DataFrame, lat_col: str, lon_col: str, tag_col: str):
    d = df.copy()
    # case-insensitive membership
    d["_is_green"] = d[tag_col].astype(str).str.contains(r"(^|;)\s*Green_Flag\s*(;|$)", case=False, na=False)
    d["_is_red"] = d[tag_col].astype(str).apply(
        lambda s: any(re.search(fr"(^|;)\s*{re.escape(cat)}\s*(;|$)", str(s), flags=re.IGNORECASE) for cat in RED_SET)
    )

    greens = d[d["_is_green"] & d[lat_col].notna() & d[lon_col].notna()]
    reds = d[d["_is_red"] & d[lat_col].notna() & d[lon_col].notna()]

    layer_green = pdk.Layer(
        "ScatterplotLayer",
        data=greens,
        get_position=[lon_col, lat_col],
        get_radius=80,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=8,
        get_fill_color=[0, 180, 0, 180],
        auto_highlight=True,
    )
    layer_red = pdk.Layer(
        "ScatterplotLayer",
        data=reds,
        get_position=[lon_col, lat_col],
        get_radius=80,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=8,
        get_fill_color=[200, 30, 30, 180],
        auto_highlight=True,
    )
    return layer_green, layer_red

def category_section(category: str,
                     df: pd.DataFrame,
                     tag_col: str,
                     lat_col: str,
                     lon_col: str,
                     deo_col: str,
                     contractor_col: str,
                     cost_col: str):
    st.subheader(f" {category} — Projects, Metrics, and Map")

    cat_df = filter_by_category_inclusive(df, tag_col, category)
    st.caption(f"{len(cat_df):,} projects currently tagged (including overlaps).")

    # Per-DEO metrics table (mirrors Green Flag metrics)
    if deo_col:
        tmp = cat_df.rename(columns={
            deo_col: "DistrictEngineeringOffice",
            (cost_col or "ContractCost"): "ContractCost",
            tag_col: "Tagging"
        })
        metrics = compute_category_metrics(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging", category)
        st.markdown("##### Metrics per District Engineering Office")
        st.dataframe(metrics[[
            "DistrictEngineeringOffice",
            "total_projects",
            "category_projects",
            "total_contract_cost",
            f"{category}_contract_cost",
            "category_density",
            "category_cost_ratio"
        ]])
    else:
        st.info("DistrictEngineeringOffice column not found — cannot compute per-DEO metrics.")

    # Map
    st.markdown("##### Map")
    if lat_col and lon_col and lat_col in cat_df.columns and lon_col in cat_df.columns:
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=MAP_INITIAL_VIEW,
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=cat_df.dropna(subset=[lat_col, lon_col]),
                    get_position=[lon_col, lat_col],
                    get_radius=80,
                    pickable=True,
                    radius_min_pixels=3,
                    radius_max_pixels=8,
                    get_fill_color=[30, 60, 200, 180] if category in GREEN_SET else [220, 90, 30, 180],
                    auto_highlight=True,
                )
            ],
            tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
        ))
    else:
        st.info("Latitude/Longitude columns not found for mapping in this section.")

    # Raw table for this category
    st.markdown("##### Projects Table")
    st.dataframe(cat_df)

    # Top entities (count)
    st.markdown("##### Top District Engineering Offices & Contractors (by Project Count)")
    ents = top_entities(cat_df, deo_col, contractor_col, cost_col, by="count", topn=15)
    cols = st.columns(2)
    if "DistrictEngineeringOffice" in ents:
        with cols[0]:
            st.dataframe(ents["DistrictEngineeringOffice"])
    if "Contractor" in ents:
        with cols[1]:
            st.dataframe(ents["Contractor"])

    # Top entities (cost)
    if cost_col:
        st.markdown("##### Top District Engineering Offices & Contractors (by Total Contract Cost)")
        ents_cost = top_entities(cat_df, deo_col, contractor_col, cost_col, by="cost", topn=15)
        cols2 = st.columns(2)
        if "DistrictEngineeringOffice" in ents_cost:
            with cols2[0]:
                st.dataframe(ents_cost["DistrictEngineeringOffice"])
        if "Contractor" in ents_cost:
            with cols2[1]:
                st.dataframe(ents_cost["Contractor"])

# -------------------------------
# Sidebar — data & options
# -------------------------------
with st.sidebar:
    st.title("🛰️ FAIL Checker Dashboard")
    data_path = st.text_input("Excel data path", DATA_PATH_DEFAULT)
    with st.expander("Settings", expanded=False):
        st.write("Map options and general preferences can be adjusted here in future versions.")

# -------------------------------
# Load & prepare
# -------------------------------
try:
    df_raw = load_data(data_path)
except Exception as e:
    st.error(f"Failed to read Excel file at '{data_path}'. Error: {e}")
    st.stop()

df = normalize_columns(df_raw)

# Detect key columns
proj_col = detect_project_id_col(df)
lat_col, lon_col = detect_lat_lon_cols(df)
deo_col = detect_deo_col(df)
contractor_col = detect_contractor_col(df)
cost_col = detect_cost_col(df)
tag_col = detect_tag_col(df)

# Deduplicate by ProjectID (keep last)
if proj_col:
    df = deduplicate_by_last(df, proj_col)
else:
    st.warning("Project ID column not detected; skipping deduplication.")

# Sanity
if not tag_col:
    st.error("Could not detect the tag/category column that contains FAIL Checker labels. "
             "Please rename your tag column to include 'Tag', 'Category', or include known labels like 'Green_Flag'.")
    st.stop()

# -------------------------------
# OVERVIEW
# -------------------------------
st.header("OVERVIEW")

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("#### Tag Distribution (including overlaps)")
    exploded = explode_tags(df, tag_col)
    tag_counts = (exploded["Tag"].value_counts()
                  .reindex(CATEGORIES, fill_value=0)
                  .rename_axis("Tag")
                  .reset_index(name="Projects"))
    st.dataframe(tag_counts)

with colB:
    st.markdown("#### Ranking Mode")
    top_mode = st.radio("Top Rank By", ["Project Count", "Total Cost"])

# Maps: Green vs Red
st.markdown("#### Comparison Map — Green vs Red")
if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
    layer_green, layer_red = make_green_red_layers(df.assign(Tagging=df[tag_col]), lat_col, lon_col, tag_col)
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=MAP_INITIAL_VIEW,
        layers=[layer_green, layer_red],
        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
    ))
else:
    st.info("Latitude/Longitude columns not found; skipping Green vs Red map.")

# Top entities
st.markdown("#### Top District Engineering Offices & Contractors")
mode = "count" if top_mode == "Project Count" else "cost"
tops = top_entities(df, deo_col, contractor_col, cost_col, by=mode, topn=15)
cols = st.columns(2)
if "DistrictEngineeringOffice" in tops:
    with cols[0]:
        st.write("**District Engineering Offices**")
        st.dataframe(tops["DistrictEngineeringOffice"])
if "Contractor" in tops:
    with cols[1]:
        st.write("**Contractors**")
        st.dataframe(tops["Contractor"])

# -------------------------------
# GREEN FLAG ANALYSIS
# -------------------------------
st.header("GREEN FLAG ANALYSIS")
if deo_col:
    # Map unified column names just for metric computation
    tmp = df.rename(columns={
        deo_col: "DistrictEngineeringOffice",
        (cost_col or "ContractCost"): "ContractCost",
        tag_col: "Tagging"
    })
    green_metrics = compute_green_metrics(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging")
    st.markdown("##### Metrics per District Engineering Office")
    st.dataframe(green_metrics[[
        "DistrictEngineeringOffice",
        "total_projects",
        "green_flag_projects",
        "total_contract_cost",
        "green_flag_contract_cost",
        "green_flag_density",
        "green_flag_cost_ratio"
    ]])
else:
    st.info("DistrictEngineeringOffice column not found — cannot compute green flag metrics per DEO.")

# Green-only map
st.markdown("#### Green Flag Projects Map")
green_df = filter_by_category_inclusive(df.assign(Tagging=df[tag_col]), "Tagging", "Green_Flag")
if lat_col and lon_col and lat_col in green_df.columns and lon_col in green_df.columns:
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=MAP_INITIAL_VIEW,
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=green_df.dropna(subset=[lat_col, lon_col]),
                get_position=[lon_col, lat_col],
                get_radius=80,
                pickable=True,
                radius_min_pixels=3,
                radius_max_pixels=8,
                get_fill_color=[0, 170, 60, 180],
                auto_highlight=True,
            )
        ],
        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
    ))
else:
    st.info("Latitude/Longitude columns not found for Green Flag map.")

# -------------------------------
# CATEGORY DEEP-DIVES
# -------------------------------
st.header("CATEGORY DEEP-DIVES")
for cat in CATEGORIES:
    if cat == "Green_Flag":
        st.divider()
        continue  # already detailed above
    st.divider()
    category_section(cat,
                     df.assign(Tagging=df[tag_col]),
                     "Tagging",
                     lat_col,
                     lon_col,
                     deo_col,
                     contractor_col,
                     cost_col)

st.caption("Note: Category filters include overlapping tags, e.g., rows tagged 'Chop_chop_Project;Siyam-siyam_Project' will appear in both sections.")
