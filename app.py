
from pathlib import Path
from urllib.parse import quote
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


DATA_PATH = Path("FAIL_Checker_Model_Output.xlsx")


# -----------------------------
# Column auto-detection
# -----------------------------
def _pick(df: pd.DataFrame, aliases: list[str] | tuple[str, ...]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    return None


ALIASES = {
    # keep ProjectID and ContractID distinct
    "project_id":  ["ProjectID", "Project Id", "project_id", "ProjID"],
    "contract_id": ["ContractID", "Contract Id", "contract_id"],
    "title":       ["ProjectTitle", "Project Title", "Title"],
    "desc":        ["ProjectDescription", "Project Description", "Description", "ScopeOfWork", "Project_Scope"],
    "category":    ["Project_Category", "Category", "Tags", "ModelTag", "ProjectTag"],
    "deo":         ["DistrictEngineeringOffice", "DEO", "ImplementingOffice", "Implementing Office", "ProcuringEntity"],
    "region":      ["Region", "REGION"],
    "province":    ["Province", "PROVINCE"],
    "municipality":["Municipality", "City", "CityMunicipality", "City_Municipality", "Municipality_City"],
    "contractor":  ["Contractor", "Supplier", "Vendor"],
    "cost":        ["ContractCost", "Contract Cost", "ApprovedBudgetForTheContract", "ABC", "Amount"],
    "infra_year":  ["InfraYear", "Year"],
    "start":       ["StartDate", "Start Date"],
    "comp_orig":   ["CompletionDateOriginal", "OriginalCompletion", "Completion Date Original"],
    "comp_act":    ["CompletionDateActual", "ActualCompletion", "Completion Date Actual"],
    "lat":         ["Latitude", "Lat"],
    "lon":         ["Longitude", "Lon", "Long"],
}


@st.cache_data(show_spinner=False)
def load_data(path: Path):
    if not path.exists():
        st.error(f"Data file not found: {path.resolve()}")
        st.stop()
    df = pd.read_excel(path)

    # Auto-detect columns
    COL = {k: _pick(df, v) for k, v in ALIASES.items()}

    # Hard guards for must-have fields
    must_have = ["category", "deo"]
    for k in must_have:
        if not COL[k]:
            st.error(f"Missing required column for {k}. Tried aliases: {ALIASES[k]}")
            st.stop()

    # Ensure cost numeric (if present)
    if COL["cost"]:
        df[COL["cost"]] = pd.to_numeric(df[COL["cost"]], errors="coerce")

    return df, COL


# -----------------------------
# Helpers
# -----------------------------
def parse_categories(series: pd.Series) -> list[list[str]]:
    return [[p.strip() for p in str(v).split(";") if p.strip()] for v in series.fillna("")]


def has_cat(series: pd.Series, token: str) -> pd.Series:
    """Boundary-aware category membership: matches ';' separated tokens."""
    pat = rf"(?:^|;)\s*{re.escape(token)}\s*(?:;|$)"
    return series.fillna("").str.contains(pat, regex=True)


def compute_deo_metrics(df: pd.DataFrame, COL: dict, active_category: str) -> pd.DataFrame:
    deo_col = COL["deo"]
    cat_col = COL["category"]
    cost_col = COL["cost"]

    # Accept either Doppelganger spelling
    if active_category == "Dopplelganger_Project":
        cat_mask = has_cat(df[cat_col], "Dopplelganger_Project") | has_cat(df[cat_col], "Doppelganger_Project")
    elif active_category == "Green_Flag":
        non_green = ["Chop_chop_Project", "Dopplelganger_Project", "Doppelganger_Project", "Ghost_Project", "Siyam-siyam_Project"]
        cat_mask = ~df[cat_col].fillna("").str.contains("|".join([re.escape(x) for x in non_green]))
    else:
        cat_mask = has_cat(df[cat_col], active_category)

    grp = df.groupby(deo_col, dropna=False)
    total_projects = grp.size().rename("total_projects")
    category_projects = grp.apply(lambda g: int(cat_mask.loc[g.index].sum())).rename("category_projects")

    if cost_col:
        total_contract_cost = grp[cost_col].sum(min_count=1).rename("total_contract_cost")
        category_contract_cost = grp.apply(lambda g: g.loc[cat_mask.loc[g.index], cost_col].sum()).rename(
            f"{active_category}_contract_cost"
        )
        out = pd.concat([total_projects, category_projects, total_contract_cost, category_contract_cost], axis=1).reset_index()
        out["category_cost_ratio"] = out[f"{active_category}_contract_cost"] / out["total_contract_cost"].replace({0: pd.NA})
    else:
        out = pd.concat([total_projects, category_projects], axis=1).reset_index()
        out["total_contract_cost"] = np.nan
        out[f"{active_category}_contract_cost"] = np.nan
        out["category_cost_ratio"] = np.nan

    out["category_density"] = out["category_projects"] / out["total_projects"].replace({0: pd.NA})
    return out


def build_clickable_metrics(dm_raw: pd.DataFrame, COL: dict, active_category: str) -> pd.DataFrame:
    dm = dm_raw.copy()
    deo_col = COL["deo"]

    # Create label text with thousands separators but keep raw value for URL
    if "category_projects" in dm.columns:
        dm["category_projects_label"] = dm["category_projects"].astype("Int64").map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
        dm["category_projects"] = dm.apply(
            lambda r: f"?drill=1&deo={quote(str(r[deo_col]))}&cat={quote(active_category)}", axis=1
        )
    return dm


def stacked_by_deo_chart(df: pd.DataFrame, COL: dict, active_category: str):
    deo = COL["deo"]
    cat = COL["category"]

    non_green = ["Chop_chop_Project", "Dopplelganger_Project", "Doppelganger_Project", "Ghost_Project", "Siyam-siyam_Project"]

    if active_category == "Green_Flag":
        is_active = ~df[cat].fillna("").str.contains("|".join([re.escape(x) for x in non_green]))
        active_label, other_label = "Green", "Red"
    else:
        if active_category == "Dopplelganger_Project":
            is_active = has_cat(df[cat], "Dopplelganger_Project") | has_cat(df[cat], "Doppelganger_Project")
        else:
            is_active = has_cat(df[cat], active_category)
        active_label, other_label = active_category.replace("_", " ").replace("-", " "), "Others"

    tmp = df[[deo]].copy()
    tmp["bucket"] = np.where(is_active, active_label, other_label)
    agg = tmp.groupby([deo, "bucket"]).size().rename("count").reset_index()

    totals = agg.groupby(deo)["count"].sum().reset_index().rename(columns={"count": "total"})
    data = agg.merge(totals, on=deo, how="left")

    if active_category == "Green_Flag":
        scale = alt.Scale(domain=[active_label, other_label], range=["#2ecc71", "#e74c3c"])
    elif active_category == "Ghost_Project":
        scale = alt.Scale(domain=[active_label, other_label], range=["#e74c3c", "#bdc3c7"])
    elif active_category == "Chop_chop_Project":
        scale = alt.Scale(domain=[active_label, other_label], range=["#e67e22", "#bdc3c7"])
    elif active_category in ("Dopplelganger_Project", "Doppelganger_Project"):
        scale = alt.Scale(domain=[active_label, other_label], range=["#9b59b6", "#bdc3c7"])
    else:  # Siyam-siyam
        scale = alt.Scale(domain=[active_label, other_label], range=["#f1c40f", "#bdc3c7"])

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", stack="zero", title="Projects"),
            y=alt.Y(f"{deo}:N", sort=alt.SortField(field="total", order="descending"), title="District Engineering Office"),
            color=alt.Color("bucket:N", scale=scale, title=""),
            tooltip=[
                alt.Tooltip(f"{deo}:N", title="DEO"),
                alt.Tooltip("bucket:N", title="Group"),
                alt.Tooltip("count:Q", title="Projects", format=",d"),
                alt.Tooltip("total:Q", title="Total (DEO)", format=",d"),
            ],
        )
        .properties(height=480, width="container")
    )
    return chart


def show_drilldown(df: pd.DataFrame, COL: dict):
    params = st.experimental_get_query_params()
    if params.get("drill") != ["1"]:
        return

    deo_col = COL["deo"]
    cat_col = COL["category"]
    sel_deo = params.get("deo", [""])[0]
    sel_cat = params.get("cat", [""])[0]

    if sel_cat == "Green_Flag":
        non_green = ["Chop_chop_Project", "Dopplelganger_Project", "Doppelganger_Project", "Ghost_Project", "Siyam-siyam_Project"]
        cat_mask = ~df[cat_col].fillna("").str.contains("|".join([re.escape(x) for x in non_green]))
    elif sel_cat == "Dopplelganger_Project":
        cat_mask = has_cat(df[cat_col], "Dopplelganger_Project") | has_cat(df[cat_col], "Doppelganger_Project")
    else:
        cat_mask = has_cat(df[cat_col], sel_cat)

    deo_mask = df[deo_col].astype(str) == str(sel_deo)
    details = df.loc[deo_mask & cat_mask].copy()

    st.subheader("Project Details")
    st.caption(f"DEO: {sel_deo} | Category: {sel_cat}")

    # Column order: show ProjectID and ContractID separately if both exist
    preferred = [c for c in [COL.get("project_id"), COL.get("contract_id")] if c] + [
        COL.get("title") or "", COL.get("desc") or "", COL.get("contractor") or "",
        deo_col, COL.get("region") or "", COL.get("province") or "", COL.get("municipality") or "",
        COL.get("infra_year") or "", COL.get("cost") or "",
        COL.get("start") or "", COL.get("comp_orig") or "", COL.get("comp_act") or "",
        COL["category"], COL.get("lat") or "", COL.get("lon") or "",
    ]
    preferred = [c for c in preferred if c]  # drop blanks
    show_cols = [c for c in preferred if c in details.columns] + [c for c in details.columns if c not in preferred]
    details = details[show_cols]

    # Formatting (if those columns exist)
    col_config = {}
    if COL.get("cost") and COL["cost"] in details.columns:
        col_config[COL["cost"]] = st.column_config.NumberColumn("Contract Cost", format="%,.2f")
    if COL.get("infra_year") and COL["infra_year"] in details.columns:
        col_config[COL["infra_year"]] = st.column_config.NumberColumn("Infra Year", format="%,d")

    st.dataframe(details, use_container_width=True, hide_index=True, column_config=col_config)

    if st.button("Back to summary"):
        st.experimental_set_query_params()
        st.experimental_rerun()


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="FAIL Checker — DEO Metrics", layout="wide")
st.title("FAIL Checker — Per DEO Metrics")

df, COL = load_data(DATA_PATH)

# Sidebar: choose category from actual data (with fallback defaults)
detected = set()
for cats in parse_categories(df[COL["category"]]):
    detected.update(cats)

# normalize Doppelganger option presence
if "Doppelganger_Project" in detected and "Dopplelganger_Project" not in detected:
    detected.add("Dopplelganger_Project")

default_categories = [
    "Green_Flag",
    "Chop_chop_Project",
    "Dopplelganger_Project",
    "Ghost_Project",
    "Siyam-siyam_Project",
]

options = sorted(detected) if detected else default_categories
default_idx = options.index("Green_Flag") if "Green_Flag" in options else 0
active_category = st.sidebar.selectbox("Category", options, index=default_idx)

# Optional filter: InfraYear (if present)
if COL.get("infra_year") and COL["infra_year"] in df.columns:
    yrs = sorted([y for y in df[COL["infra_year"]].dropna().unique().tolist() if str(y).strip() != ""])
    if yrs:
        sel_yrs = st.sidebar.multiselect("Infra Year", yrs, default=yrs)
        if sel_yrs:
            df = df[df[COL["infra_year"]].isin(sel_yrs)]

# Compute metrics
with st.spinner("Computing metrics..."):
    metrics = compute_deo_metrics(df, COL, active_category)
    metrics = metrics.sort_values(["category_density", "category_projects"], ascending=[False, False])

# Render metrics table (commas + clickable category_projects)
st.subheader("Per DEO Metrics")
dm = build_clickable_metrics(metrics, COL, active_category)

# Column config with thousands formatting
table_config = {
    COL["deo"]: st.column_config.TextColumn("District Engineering Office"),
    "total_projects": st.column_config.NumberColumn("Total Projects", format="%,d"),
    "category_projects": st.column_config.LinkColumn(
        "Category Projects",
        display_text="${category_projects_label}",
        help="Click to view projects for this DEO and category",
    ),
    "category_density": st.column_config.NumberColumn("Category Density", format=",.4f"),
    "category_cost_ratio": st.column_config.NumberColumn("Category Cost Ratio", format=",.4f"),
}
if f"{active_category}_contract_cost" in dm.columns:
    table_config[f"{active_category}_contract_cost"] = st.column_config.NumberColumn(
        f"{active_category} Contract Cost", format="%,.2f"
    )
if "total_contract_cost" in dm.columns:
    table_config["total_contract_cost"] = st.column_config.NumberColumn("Total Contract Cost", format="%,.2f")
# hide helper
table_config["category_projects_label"] = None

st.data_editor(dm, use_container_width=True, hide_index=True, column_config=table_config, key="deo_metrics_editor")

st.divider()

# Stacked chart (keeps the same spot as your former per-category bar)
st.subheader("Per DEO — Stacked Projects")
st.altair_chart(stacked_by_deo_chart(df, COL, active_category), use_container_width=True)

st.divider()

# Drilldown section (shown only after click)
show_drilldown(df, COL)
