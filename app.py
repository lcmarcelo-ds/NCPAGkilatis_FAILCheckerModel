# app.py
# ============================================
# FAIL Checker Dashboard — Per DEO Metrics with Drilldown
# - Loads FAIL_Checker_Model_Output.xlsx
# - Shows per-DEO metrics for a chosen category
# - Formats numbers with thousands separators
# - category_projects is clickable; clicking reveals full project details
# ============================================

import re
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Basic Config
# -----------------------------
DATA_PATH = Path("FAIL_Checker_Model_Output.xlsx")

# Column mapping (change here only if your column names differ)
COLS = {
    "project_id":            "ProjectID",
    "project_title":         "ProjectTitle",
    "project_desc":          "ProjectDescription",
    "category":              "Project_Category",          # multi-valued; e.g., "A;B"
    "deo":                   "DistrictEngineeringOffice", # DEO column
    "region":                "Region",
    "province":              "Province",
    "municipality":          "Municipality",
    "contractor":            "Contractor",
    "contract_cost":         "ContractCost",
    "infra_year":            "InfraYear",
    "start_date":            "StartDate",
    "comp_orig":             "CompletionDateOriginal",
    "comp_actual":           "CompletionDateActual",
    "lat":                   "Latitude",
    "lon":                   "Longitude",
}

DEFAULT_CATEGORIES = [
    "Green_Flag",
    "Chop_chop_Project",
    "Dopplelganger_Project",  # keep your spelling
    "Ghost_Project",
    "Siyam-siyam_Project",
    # If your data sometimes uses "Doppelganger_Project", we’ll also handle it below.
]

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Data file not found at: {path.resolve()}")
        st.stop()
    df = pd.read_excel(path)
    # Ensure expected columns exist
    for key, col in COLS.items():
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
    # Normalize types where helpful
    if COLS["contract_cost"] in df.columns:
        df[COLS["contract_cost"]] = pd.to_numeric(df[COLS["contract_cost"]], errors="coerce")
    return df

def parse_categories(series: pd.Series):
    """Return list-of-lists of categories per row (split on ';')."""
    out = []
    for v in series.fillna(""):
        parts = [p.strip() for p in str(v).split(";") if p.strip()]
        out.append(parts)
    return out

def has_category(series: pd.Series, target: str) -> pd.Series:
    """
    Element-wise boolean mask: Project_Category contains target token,
    respecting ';' boundaries and overlapping tags.
    """
    pat = rf"(?:^|;)\s*{re.escape(target)}\s*(?:;|$)"
    return series.fillna("").str.contains(pat, regex=True)

def compute_deo_metrics(df: pd.DataFrame, deo_col: str, category_key: str) -> pd.DataFrame:
    """
    Returns a per-DEO metrics frame:
      total_projects
      category_projects
      total_contract_cost
      {category_key}_contract_cost
      category_density (share of projects in category)
      category_cost_ratio (share of contract cost in category)
    """
    # Support both "Dopplelganger" and "Doppelganger"
    alt_key = "Doppelganger_Project" if category_key == "Dopplelganger_Project" else category_key

    cat_mask = has_category(df[COLS["category"]], category_key) | has_category(df[COLS["category"]], alt_key)

    grp = df.groupby(deo_col, dropna=False)
    total_projects = grp.size().rename("total_projects")
    category_projects = grp.apply(lambda g: int(cat_mask.loc[g.index].sum())).rename("category_projects")

    cost_col = COLS["contract_cost"]
    total_contract_cost = grp[cost_col].sum(min_count=1).rename("total_contract_cost")
    category_contract_cost = grp.apply(lambda g: g.loc[cat_mask.loc[g.index], cost_col].sum()).rename(f"{category_key}_contract_cost")

    out = pd.concat([total_projects, category_projects, total_contract_cost, category_contract_cost], axis=1).reset_index()

    # Derived metrics
    out["category_density"]   = out["category_projects"]   / out["total_projects"].replace({0: pd.NA})
    out["category_cost_ratio"] = out[f"{category_key}_contract_cost"] / out["total_contract_cost"].replace({0: pd.NA})
    return out

def build_deo_metrics_table(df_metrics: pd.DataFrame, deo_col: str, category_key: str) -> pd.DataFrame:
    """
    - Adds thousands separators via Streamlit column_config (we keep raw numbers)
    - Replaces category_projects with an app-internal link (query params)
    - Adds a hidden label column used as the link's visible text
    """
    dm = df_metrics.copy()

    rename_map = {
        "total_projects": "Total Projects",
        "category_projects": "Category Projects",
        "total_contract_cost": "Total Contract Cost",
        f"{category_key}_contract_cost": f"{category_key} Contract Cost",
        "category_density": "Category Density",
        "category_cost_ratio": "Category Cost Ratio",
    }
    dm = dm.rename(columns={k: v for k, v in rename_map.items() if k in dm.columns})

    # Build a pretty label for the link text while keeping the raw URL in the cell value
    if "Category Projects" in dm.columns:
        dm["Category Projects (label)"] = dm["Category Projects"].astype("Int64").map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
        dm["Category Projects"] = dm.apply(
            lambda r: f"?drill=1&deo={quote(str(r[deo_col]))}&cat={quote(category_key)}", axis=1
        )
    return dm

def show_metrics_table(dm: pd.DataFrame, deo_col: str, category_key: str):
    st.data_editor(
        dm,
        use_container_width=True,
        hide_index=True,
        column_config={
            deo_col: st.column_config.TextColumn("District Engineering Office"),
            # Clickable link that uses the formatted label
            "Category Projects": st.column_config.LinkColumn(
                "Category Projects",
                display_text="${Category Projects (label)}",
                help="Click to view projects for this DEO and category"
            ),
            "Total Projects": st.column_config.NumberColumn("Total Projects", format="%,d"),
            "Total Contract Cost": st.column_config.NumberColumn("Total Contract Cost", format="%,.2f"),
            f"{category_key} Contract Cost": st.column_config.NumberColumn(
                f"{category_key} Contract Cost", format="%,.2f"
            ),
            "Category Density": st.column_config.NumberColumn("Category Density", format=",.4f"),
            "Category Cost Ratio": st.column_config.NumberColumn("Category Cost Ratio", format=",.4f"),
            "Category Projects (label)": None,  # hide helper
        },
        key="deo_metrics_editor",
    )

def show_drilldown(df: pd.DataFrame, deo_col: str, category_key: str):
    params = st.experimental_get_query_params()
    if params.get("drill") != ["1"]:
        return

    sel_deo = params.get("deo", [""])[0]
    sel_cat = params.get("cat", [""])[0]

    # Support both spellings for Doppelganger
    sel_alt = "Doppelganger_Project" if sel_cat == "Dopplelganger_Project" else sel_cat

    st.markdown(f"### Project Details")
    st.markdown(f"**DEO:** {sel_deo} &nbsp;&nbsp; | &nbsp;&nbsp; **Category:** {sel_cat}")

    cat_mask = has_category(df[COLS["category"]], sel_cat) | has_category(df[COLS["category"]], sel_alt)
    deo_mask = df[deo_col].astype(str) == str(sel_deo)
    details = df.loc[deo_mask & cat_mask].copy()

    # Order useful columns first; keep the rest too
    preferred = [
        COLS["project_id"], COLS["project_title"], COLS["project_desc"],
        COLS["contractor"], deo_col, COLS["region"], COLS["province"], COLS["municipality"],
        COLS["infra_year"], COLS["contract_cost"],
        COLS["start_date"], COLS["comp_orig"], COLS["comp_actual"],
        COLS["category"], COLS["lat"], COLS["lon"]
    ]
    show_cols = [c for c in preferred if c in details.columns] + [c for c in details.columns if c not in preferred]
    details = details[show_cols]

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
        column_config={
            COLS["contract_cost"]: st.column_config.NumberColumn("Contract Cost", format="%,.2f"),
            COLS["infra_year"]: st.column_config.NumberColumn("Infra Year", format="%,d"),
        },
    )

    # Back button clears query params
    if st.button("Back to summary"):
        st.experimental_set_query_params()
        st.experimental_rerun()

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="FAIL Checker — Per DEO Metrics", layout="wide")
st.title("FAIL Checker — Per DEO Metrics with Drilldown")

df = load_data(DATA_PATH)
deo_col = COLS["deo"]

# Sidebar: category selector (auto-detect options from data; fall back to defaults)
detected_cats = set()
for cats in parse_categories(df[COLS["category"]]):
    detected_cats.update(cats)
# Normalize Doppelganger spelling in options
if "Doppelganger_Project" in detected_cats and "Dopplelganger_Project" not in detected_cats:
    detected_cats.add("Dopplelganger_Project")
cats_sorted = sorted(detected_cats) if detected_cats else DEFAULT_CATEGORIES

active_category = st.sidebar.selectbox("Category", cats_sorted, index=cats_sorted.index("Green_Flag") if "Green_Flag" in cats_sorted else 0)

# Optional: filters (e.g., InfraYear)
if COLS["infra_year"] in df.columns:
    yrs = sorted(df[COLS["infra_year"]].dropna().unique().tolist())
    if yrs:
        yr_pick = st.sidebar.multiselect("Infra Year (optional filter)", yrs, default=yrs)
        if yr_pick:
            df = df[df[COLS["infra_year"]].isin(yr_pick)]

# Compute
with st.spinner("Computing metrics…"):
    metrics = compute_deo_metrics(df, deo_col=deo_col, category_key=active_category)
    # Sort by category density then count
    metrics = metrics.sort_values(["category_density", "category_projects"], ascending=[False, False])

# Show table (formatted, clickable)
st.subheader("Per DEO Metrics")
dm = build_deo_metrics_table(metrics, deo_col=deo_col, category_key=active_category)
show_metrics_table(dm, deo_col=deo_col, category_key=active_category)

# If a count was clicked, show the full drilldown table
st.divider()
show_drilldown(df, deo_col=deo_col, category_key=active_category)
