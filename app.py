
import re
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="FAIL Checker Dashboard", layout="wide", initial_sidebar_state="expanded")

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

# deterministic category colors (RGB A)
CAT_COLOR = {
    "Green_Flag": [18, 152, 66, 190],
    "Siyam-siyam_Project": [214, 139, 0, 190],
    "Chop_chop_Project":   [189, 28, 28, 190],
    "Dopplelganger_Project":[128, 60, 170, 190],
    "Ghost_Project":       [32, 96, 168, 190],
}
MAP_INITIAL_VIEW = pdk.ViewState(latitude=12.8797, longitude=121.7740, zoom=5, pitch=0)

# ------------------------------- Utilities
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # unify column names (keep original values intact)
    new_cols = {}
    for c in df.columns:
        nc = re.sub(r"\s+", "_", str(c)).strip("_")
        new_cols[c] = nc
    return df.rename(columns=new_cols)

def find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in cols:
        lc = c.lower()
        if any(k.lower() in lc for k in candidates):
            return c
    return None

def detect_project_id_col(df):        return find_column(df, ["ProjectID","Project_Id","Project_ID","ProjID","ProjectCode","Project_Code"])
def detect_lat_lon_cols(df):          return find_column(df, ["Latitude","Lat"]), find_column(df, ["Longitude","Lon","Lng"])
def detect_deo_col(df):               return find_column(df, ["DistrictEngineeringOffice","District_Engineering_Office","DEO"])
def detect_contractor_col(df):        return find_column(df, ["Contractor","ContractorName","Supplier","SupplierName"])
def detect_year_col(df):              return find_column(df, ["Year","Start_Year","StartYear","CompletionYear","End_Year","EndYear"])
def detect_cost_col(df):
    col = find_column(df, ["ContractCost","Total_Contract_Cost","Contract_Amount","ProjectCost","Approved_Budget_for_the_Contract","ABC"])
    if col: return col
    for c in df.columns:
        lc = c.lower()
        if ("contract" in lc or "project" in lc) and ("cost" in lc or "amount" in lc or "value" in lc):
            return c
    # fallback: first numeric column with large magnitude
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def detect_tag_col(df):
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(300).str.lower()
        if vals.empty: 
            continue
        if any(any(cat.lower() in v for v in vals) for cat in CATEGORIES):
            return c
    return find_column(df, ["Tags","Tagging","FAIL_Tags","FAIL_Tag","Category","Categories","Labels"])

def ensure_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0)

def deduplicate_by_last(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
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

def add_year_if_missing(df: pd.DataFrame, year_col: str | None) -> tuple[pd.DataFrame, str | None]:
    if year_col:
        return df, year_col
    # try to derive from dates if present
    date_candidates = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
    y = None
    for c in date_candidates:
        try:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().sum() > 0:
                y = d.dt.year
                df = df.copy()
                df["__Year"] = y
                return df, "__Year"
        except Exception:
            continue
    return df, None

def kpi_card(label: str, value, help_text: str = ""):
    # simple, clean KPI using columns
    st.metric(label, value, help=help_text)

def alt_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, sort_col: str | None = None, topn: int | None = None):
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

def green_metrics_per_deo(df: pd.DataFrame, deo_col: str, cost_col: str, tag_col: str) -> pd.DataFrame:
    w = df.copy()
    w["_is_green"] = w[tag_col].astype(str).apply(lambda s: contains_category(s, "Green_Flag"))
    w["_cost"] = ensure_numeric(w[cost_col]) if cost_col else 0
    total = w.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    green_ct = w.groupby(deo_col, dropna=False)["_is_green"].sum().astype(int).reset_index(name="green_flag_projects")
    if cost_col:
        total_cost = w.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        green_cost = w[w["_is_green"]].groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="green_flag_contract_cost")
    else:
        total_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "total_contract_cost": 0})
        green_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "green_flag_contract_cost": 0})
    out = (
        total.merge(green_ct, on=deo_col, how="left")
             .merge(total_cost, on=deo_col, how="left")
             .merge(green_cost, on=deo_col, how="left")
    ).fillna({"green_flag_projects": 0, "green_flag_contract_cost": 0, "total_contract_cost": 0})
    out["green_flag_projects"] = out["green_flag_projects"].astype(int)
    out["green_flag_density"] = np.where(out["total_projects"] > 0, out["green_flag_projects"]/out["total_projects"], 0.0)
    out["green_flag_cost_ratio"] = np.where(out["total_contract_cost"] > 0, out["green_flag_contract_cost"]/out["total_contract_cost"], 0.0)
    return out.sort_values(["green_flag_projects","green_flag_density"], ascending=[False, False]).reset_index(drop=True)

def category_metrics_per_deo(df: pd.DataFrame, deo_col: str, cost_col: str, tag_col: str, category: str) -> pd.DataFrame:
    w = df.copy()
    w["_in_cat"] = w[tag_col].astype(str).apply(lambda s: contains_category(s, category))
    w["_cost"] = ensure_numeric(w[cost_col]) if cost_col else 0
    total = w.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    cat_ct = w.groupby(deo_col, dropna=False)["_in_cat"].sum().astype(int).reset_index(name="category_projects")
    if cost_col:
        total_cost = w.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        cat_cost = w[w["_in_cat"]].groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name=f"{category}_contract_cost")
    else:
        total_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "total_contract_cost": 0})
        cat_cost = pd.DataFrame({deo_col: w[deo_col].unique(), f"{category}_contract_cost": 0})
    out = (
        total.merge(cat_ct, on=deo_col, how="left")
             .merge(total_cost, on=deo_col, how="left")
             .merge(cat_cost, on=deo_col, how="left")
    ).fillna({"category_projects": 0, f"{category}_contract_cost": 0, "total_contract_cost": 0})
    out["category_projects"] = out["category_projects"].astype(int)
    out["category_density"] = np.where(out["total_projects"] > 0, out["category_projects"]/out["total_projects"], 0.0)
    out["category_cost_ratio"] = np.where(out["total_contract_cost"] > 0, out[f"{category}_contract_cost"]/out["total_contract_cost"], 0.0)
    return out.sort_values(["category_projects","category_density"], ascending=[False, False]).reset_index(drop=True)

def top_entities(df: pd.DataFrame, deo_col: str, contractor_col: str, cost_col: str | None, by="count", topn=15):
    out = {}
    if deo_col and deo_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = df.groupby(deo_col, dropna=False)[cost_col].apply(ensure_numeric).sum().sort_values(ascending=False).head(topn).reset_index(name="Total Cost")
        else:
            tmp = df.groupby(deo_col, dropna=False).size().sort_values(ascending=False).head(topn).reset_index(name="Projects")
        out["DEO"] = tmp
    if contractor_col and contractor_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = df.groupby(contractor_col, dropna=False)[cost_col].apply(ensure_numeric).sum().sort_values(ascending=False).head(topn).reset_index(name="Total Cost")
        else:
            tmp = df.groupby(contractor_col, dropna=False).size().sort_values(ascending=False).head(topn).reset_index(name="Projects")
        out["Contractor"] = tmp
    return out

def map_layers_for_categories(df: pd.DataFrame, lat_col: str, lon_col: str, tag_col: str, cats: list[str]):
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

def download_button_df(df: pd.DataFrame, label: str, file_name: str):
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv")

# ------------------------------- Sidebar: data + column mapping + filters
with st.sidebar:
    st.title("FAIL Checker Dashboard")
    data_path = st.text_input("Excel data path", DATA_PATH_DEFAULT)

    try:
        df_raw = load_data(data_path)
        load_err = None
    except Exception as e:
        load_err = str(e)

    if load_err:
        st.error(f"Failed to read Excel file: {load_err}")
        st.stop()

    df = normalize_columns(df_raw)

    # Detect columns
    colmap = {}
    colmap["project_id"]  = detect_project_id_col(df)
    colmap["latitude"]    = detect_lat_lon_cols(df)[0]
    colmap["longitude"]   = detect_lat_lon_cols(df)[1]
    colmap["deo"]         = detect_deo_col(df)
    colmap["contractor"]  = detect_contractor_col(df)
    colmap["year"]        = detect_year_col(df)
    colmap["cost"]        = detect_cost_col(df)
    colmap["tag"]         = detect_tag_col(df)

    st.markdown("Column Mapping")
    c1, c2 = st.columns(2)
    with c1:
        colmap["project_id"] = st.selectbox("Project ID", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["project_id"]) if colmap["project_id"] in df.columns else 0)
        colmap["deo"]        = st.selectbox("DistrictEngineeringOffice", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["deo"]) if colmap["deo"] in df.columns else 0)
        colmap["tag"]        = st.selectbox("Tag/Category Column", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["tag"]) if colmap["tag"] in df.columns else 0)
    with c2:
        colmap["latitude"]   = st.selectbox("Latitude", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["latitude"]) if colmap["latitude"] in df.columns else 0)
        colmap["longitude"]  = st.selectbox("Longitude", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["longitude"]) if colmap["longitude"] in df.columns else 0)
        colmap["contractor"] = st.selectbox("Contractor", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["contractor"]) if colmap["contractor"] in df.columns else 0)
    colmap["cost"] = st.selectbox("Contract Cost", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["cost"]) if colmap["cost"] in df.columns else 0)
    colmap["year"] = st.selectbox("Year", options=[""] + list(df.columns), index=([""]+list(df.columns)).index(colmap["year"]) if colmap["year"] in df.columns else 0)

# Apply deduplication
proj_col = colmap["project_id"] if colmap["project_id"] else None
df = deduplicate_by_last(df, proj_col)

# Add derived year if missing
df, derived_year_col = add_year_if_missing(df, colmap["year"])
year_col = colmap["year"] or derived_year_col

# Validate tag col
tag_col = colmap["tag"]
if not tag_col:
    st.error("A tag/category column is required (containing labels like 'Green_Flag', 'Chop_chop_Project', etc.).")
    st.stop()

# ------------------------------- Global Filters
st.sidebar.markdown("---")
st.sidebar.markdown("Global Filters")

working = df.copy()

# Filter by year (if available)
if year_col:
    years = sorted(set(working[year_col].dropna().astype(int)))
    ysel = st.sidebar.select_slider("Year range", options=years, value=(years[0], years[-1]))
    working = working[(working[year_col].astype(float) >= ysel[0]) & (working[year_col].astype(float) <= ysel[1])]

# Filter by DEO
deo_col = colmap["deo"]
if deo_col:
    deo_opts = sorted([x for x in working[deo_col].dropna().astype(str).unique()])
    deo_sel = st.sidebar.multiselect("District Engineering Office", options=deo_opts, default=deo_opts)
    if deo_sel:
        working = working[working[deo_col].astype(str).isin(deo_sel)]

# Filter by Contractor
contractor_col = colmap["contractor"]
if contractor_col:
    ctr_opts = sorted([x for x in working[contractor_col].dropna().astype(str).unique()])
    ctr_sel = st.sidebar.multiselect("Contractor", options=ctr_opts, default=ctr_opts)
    if ctr_sel:
        working = working[working[contractor_col].astype(str).isin(ctr_sel)]

# ------------------------------- Tabs
tabs = st.tabs(["Overview", "Compare", "Green Flag", "Siyam-siyam", "Chop-chop", "Dopplelganger", "Ghost"])

# ===== Overview =====
with tabs[0]:
    st.header("Overview")

    # KPIs
    total_projects = len(working)
    exploded = explode_tags(working, tag_col)
    tag_counts = exploded["Tag"].value_counts().reindex(CATEGORIES, fill_value=0)
    green_count = int(tag_counts.get("Green_Flag", 0))
    red_count = int(tag_counts.reindex(list(RED_SET), fill_value=0).sum())
    green_share = (green_count / total_projects * 100) if total_projects > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects (after filters)", f"{total_projects:,}")
    c2.metric("Green Flag Projects", f"{green_count:,}")
    c3.metric("Red-tagged Projects", f"{red_count:,}")
    c4.metric("Green Share (%)", f"{green_share:.2f}")

    # Distribution chart
    dist_df = tag_counts.rename_axis("Tag").reset_index(name="Projects")
    alt_bar(dist_df, x_col="Tag", y_col="Projects", title="Distribution of Tags", sort_col="Projects")

    # Map: Green vs Red
    st.subheader("Comparison Map — Green vs Red")
    lat_col, lon_col = colmap["latitude"], colmap["longitude"]
    if lat_col and lon_col and lat_col in working.columns and lon_col in working.columns:
        # Two layers only (green + red)
        d = working.copy()
        d["_is_green"] = d[tag_col].astype(str).apply(lambda s: contains_category(s, "Green_Flag"))
        d["_is_red"]   = d[tag_col].astype(str).apply(lambda s: any(contains_category(s, c) for c in RED_SET))
        greens = d[d["_is_green"] & d[lat_col].notna() & d[lon_col].notna()]
        reds   = d[d["_is_red"]   & d[lat_col].notna() & d[lon_col].notna()]

        deck = pdk.Deck(
            map_style=None,
            initial_view_state=MAP_INITIAL_VIEW,
            layers=[
                pdk.Layer("ScatterplotLayer", data=greens, get_position=[lon_col, lat_col],
                          get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                          get_fill_color=CAT_COLOR["Green_Flag"], auto_highlight=True),
                pdk.Layer("ScatterplotLayer", data=reds, get_position=[lon_col, lat_col],
                          get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                          get_fill_color=[200, 40, 40, 190], auto_highlight=True),
            ],
            tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
        )
        st.pydeck_chart(deck)
    else:
        st.info("Latitude/Longitude columns are required for the map.")

    # Top entities
    st.subheader("Top Entities")
    cost_col = colmap["cost"]
    mode = st.radio("Rank by", ["Project Count", "Total Cost"], horizontal=True, index=0)
    mode_key = "count" if mode == "Project Count" else "cost"
    tops = top_entities(working, deo_col, contractor_col, cost_col, by=mode_key, topn=15)
    t1, t2 = st.columns(2)
    if "DEO" in tops: t1.dataframe(tops["DEO"])
    if "Contractor" in tops: t2.dataframe(tops["Contractor"])

    st.subheader("Filtered Data")
    st.dataframe(working)
    download_button_df(working, "Download filtered CSV", "filtered_projects.csv")

# ===== Compare =====
with tabs[1]:
    st.header("Category Comparison Map")
    lat_col, lon_col = colmap["latitude"], colmap["longitude"]
    if lat_col and lon_col and lat_col in working.columns and lon_col in working.columns:
        selected_cats = st.multiselect("Select categories to map", options=CATEGORIES, default=CATEGORIES)
        layers = map_layers_for_categories(working, lat_col, lon_col, tag_col, selected_cats)
        deck = pdk.Deck(map_style=None, initial_view_state=MAP_INITIAL_VIEW, layers=layers,
                        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"})
        st.pydeck_chart(deck)
    else:
        st.info("Latitude/Longitude columns are required for the map.")

    st.subheader("Category Counts")
    dist_df = explode_tags(working, tag_col)["Tag"].value_counts().reindex(CATEGORIES, fill_value=0).rename_axis("Tag").reset_index(name="Projects")
    st.dataframe(dist_df)
    alt_bar(dist_df, x_col="Tag", y_col="Projects", title="Counts by Category", sort_col="Projects")

# ===== Green Flag =====
with tabs[2]:
    st.header("Green Flag Analysis")
    if deo_col:
        tmp = working.rename(columns={deo_col: "DistrictEngineeringOffice", (cost_col or "ContractCost"): "ContractCost", tag_col: "Tagging"})
        gm = green_metrics_per_deo(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging")
        st.subheader("Per-DEO Metrics")
        st.dataframe(gm[["DistrictEngineeringOffice","total_projects","green_flag_projects","total_contract_cost","green_flag_contract_cost","green_flag_density","green_flag_cost_ratio"]])
        alt_bar(gm.rename(columns={"DistrictEngineeringOffice":"DEO","green_flag_projects":"Projects"}),
                x_col="DEO", y_col="Projects", title="Green Flag Projects by DEO", sort_col="Projects", topn=25)
    else:
        st.info("DistrictEngineeringOffice column is required for per-DEO metrics.")

    st.subheader("Map")
    if lat_col and lon_col and lat_col in working.columns and lon_col in working.columns:
        gdf = filter_inclusive(working.assign(Tagging=working[tag_col]), "Tagging", "Green_Flag")
        deck = pdk.Deck(map_style=None, initial_view_state=MAP_INITIAL_VIEW,
                        layers=[pdk.Layer("ScatterplotLayer", data=gdf.dropna(subset=[lat_col,lon_col]),
                                          get_position=[lon_col,lat_col], get_radius=80, pickable=True,
                                          radius_min_pixels=3, radius_max_pixels=8, get_fill_color=CAT_COLOR["Green_Flag"], auto_highlight=True)],
                        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"})
        st.pydeck_chart(deck)
    else:
        st.info("Latitude/Longitude columns are required for the map.")

    st.subheader("Table")
    st.dataframe(filter_inclusive(working, tag_col, "Green_Flag"))

# ===== Helper to render a category tab
def render_category_tab(df_in: pd.DataFrame, category: str, tab_container):
    with tab_container:
        st.header(category.replace("_"," "))
        if deo_col:
            tmp = df_in.rename(columns={deo_col: "DistrictEngineeringOffice", (cost_col or "ContractCost"): "ContractCost", tag_col: "Tagging"})
            cm = category_metrics_per_deo(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging", category)
            st.subheader("Per-DEO Metrics")
            cols = ["DistrictEngineeringOffice","total_projects","category_projects","total_contract_cost",f"{category}_contract_cost","category_density","category_cost_ratio"]
            st.dataframe(cm[cols])
            alt_bar(cm.rename(columns={"DistrictEngineeringOffice":"DEO","category_projects":"Projects"}),
                    x_col="DEO", y_col="Projects", title=f"{category} Projects by DEO", sort_col="Projects", topn=25)
        else:
            st.info("DistrictEngineeringOffice column is required for per-DEO metrics.")

        st.subheader("Map")
        if colmap["latitude"] and colmap["longitude"] and colmap["latitude"] in df_in.columns and colmap["longitude"] in df_in.columns:
            cdf = filter_inclusive(df_in.assign(Tagging=df_in[tag_col]), "Tagging", category)
            deck = pdk.Deck(map_style=None, initial_view_state=MAP_INITIAL_VIEW,
                            layers=[pdk.Layer("ScatterplotLayer", data=cdf.dropna(subset=[colmap["latitude"],colmap["longitude"]]),
                                              get_position=[colmap["longitude"], colmap["latitude"]],
                                              get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                                              get_fill_color=CAT_COLOR.get(category,[120,120,120,160]), auto_highlight=True)],
                            tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"})
            st.pydeck_chart(deck)
        else:
            st.info("Latitude/Longitude columns are required for the map.")

        st.subheader("Table")
        st.dataframe(filter_inclusive(df_in, tag_col, category))

# ===== Siyam-siyam =====
render_category_tab(working, "Siyam-siyam_Project", tabs[3])

# ===== Chop-chop =====
render_category_tab(working, "Chop_chop_Project", tabs[4])

# ===== Dopplelganger =====
render_category_tab(working, "Dopplelganger_Project", tabs[5])

# ===== Ghost =====
render_category_tab(working, "Ghost_Project", tabs[6])
