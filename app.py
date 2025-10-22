import re
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="FAIL Checker Dashboard", layout="wide", initial_sidebar_state="expanded")

# =============================================================
# CONFIG
# =============================================================
DATA_PATH_DEFAULT = "FAIL_Checker_Model_Output.xlsx"

CATEGORIES = [
    "Green_Flag",
    "Siyam-siyam_Project",
    "Chop_chop_Project",
    "Doppelganger_Project",
    "Ghost_Project",
]
RED_SET = {"Siyam-siyam_Project","Chop_chop_Project","Doppelganger_Project","Ghost_Project"}
CAT_COLOR = {
    "Green_Flag": [18,152,66,190],
    "Siyam-siyam_Project": [214,139,0,190],
    "Chop_chop_Project": [189,28,28,190],
    "Doppelganger_Project": [128,60,170,190],
    "Ghost_Project": [32,96,168,190],
}
MAP_INITIAL_VIEW = pdk.ViewState(latitude=12.8797, longitude=121.7740, zoom=5)

# =============================================================
# UTILS
# =============================================================
@st.cache_data(show_spinner=False)
def load_data(path): return pd.read_excel(path)

def normalize_columns(df):
    return df.rename(columns={c:re.sub(r"\s+","_",c.strip()):c for c in df.columns})

def find_column(df, candidates):
    for c in df.columns:
        if any(k.lower()==c.lower() for k in candidates): return c
    for c in df.columns:
        if any(k.lower() in c.lower() for k in candidates): return c
    return None

def detect(df, keys):
    return find_column(df, keys)

def ensure_numeric(s): return pd.to_numeric(s, errors="coerce").fillna(0)

def contains_category(s, cat):
    return bool(re.search(fr"(^|;)\s*{re.escape(cat)}\s*(;|$)",str(s),re.I))

def filter_inclusive(df, col, cat):
    return df[df[col].astype(str).apply(lambda s:contains_category(s,cat))].copy()

def deck_chart(layers): return pdk.Deck(initial_view_state=MAP_INITIAL_VIEW, layers=layers)

# =============================================================
# LOAD
# =============================================================
df = normalize_columns(load_data(DATA_PATH_DEFAULT))
proj_col = detect(df,["ProjectID"])
deo_col  = detect(df,["DistrictEngineeringOffice"])
lat_col  = detect(df,["Latitude"])
lon_col  = detect(df,["Longitude"])
cost_col = detect(df,["ContractCost","ApprovedBudgetForTheContract"])
tag_col  = detect(df,["Project_Category","Tagging","FAIL_Tag"])
if not all([proj_col,deo_col,lat_col,lon_col,cost_col,tag_col]):
    st.error("Missing required columns.")
    st.stop()

# =============================================================
# STACKED CHART + DRILLDOWN HELPERS
# =============================================================
def _has_cat(series, token):
    pat = rf"(?:^|;)\s*{re.escape(token)}\s*(?:;|$)"
    return series.fillna("").str.contains(pat,regex=True,case=False)

def stacked_deo_share_chart(df, deo_col, tag_col, category, topn=20):
    non_green = {"Siyam-siyam_Project","Chop_chop_Project","Doppelganger_Project","Ghost_Project"}
    if category=="Green_Flag":
        is_active = ~df[tag_col].fillna("").str.contains("|".join(map(re.escape,non_green)),case=False)
        a_label,o_label="Green","Red"
    else:
        is_active = _has_cat(df[tag_col],category)
        a_label,o_label=category.replace("_"," "), "Others"
    tmp=df[[deo_col]].copy(); tmp["bucket"]=np.where(is_active,a_label,o_label)
    agg=tmp.groupby([deo_col,"bucket"]).size().rename("count").reset_index()
    totals=agg.groupby(deo_col)["count"].sum().rename("total").reset_index()
    data=agg.merge(totals,on=deo_col); keep=totals.sort_values("total",ascending=False).head(topn)[deo_col]
    data=data[data[deo_col].isin(keep)]; data["share"]=100*data["count"]/data["total"].replace({0:np.nan})
    color_range=["#2ecc71","#e74c3c"] if category=="Green_Flag" else ["#e67e22","#bdc3c7"]
    chart=(alt.Chart(data).mark_bar().encode(
        x=alt.X("share:Q",title="Share (%)",stack="normalize"),
        y=alt.Y(f"{deo_col}:N",sort="-x",title="DEO"),
        color=alt.Color("bucket:N",scale=alt.Scale(domain=[a_label,o_label],range=color_range)),
        tooltip=[deo_col,"bucket","count","share"]
    ).properties(height=450))
    return chart

def _linkify_projects(df, deo_col, count_col, category):
    out=df.copy()
    label=f"{count_col}_label"
    out[label]=out[count_col].apply(lambda x:f"{x:,}")
    out[count_col]=out.apply(lambda r:f"?drill=1&deo={r[deo_col]}&cat={category}",axis=1)
    return out,label

def _drilldown_panel(df, deo_col, tag_col):
    params=st.query_params
    if params.get("drill")!="1": return
    sel_deo=params.get("deo"); sel_cat=params.get("cat")
    if not (sel_deo and sel_cat): return
    cat_mask=_has_cat(df[tag_col],sel_cat)
    deo_mask=df[deo_col].astype(str)==str(sel_deo)
    rows=df.loc[deo_mask & cat_mask].copy()
    st.subheader(f"Projects in {sel_deo} ({sel_cat})")
    st.dataframe(rows,use_container_width=True,hide_index=True)

# =============================================================
# MAIN UI
# =============================================================
tabs=st.tabs(["Overview","Green Flag","Siyam-siyam","Chop-chop","Doppelganger","Ghost"])

# ========== OVERVIEW ==========
with tabs[0]:
    st.header("Overview")
    total=len(df)
    exploded=(df.assign(Tag=df[tag_col].astype(str).str.split(";")).explode("Tag"))
    counts=exploded["Tag"].value_counts().reindex(CATEGORIES,fill_value=0)
    c1,c2,c3,c4=st.columns(4)
    green=int(counts["Green_Flag"]); red=int(counts[list(RED_SET)].sum())
    c1.metric("Projects",f"{total:,}")
    c2.metric("Green",f"{green:,}")
    c3.metric("Red",f"{red:,}")
    c4.metric("Green Share",f"{(green/total*100 if total else 0):.1f}%")

# ========== GREEN FLAG ==========
with tabs[1]:
    st.header("Green Flag Analysis")
    deo_grp=df.groupby(deo_col,dropna=False)
    gm=pd.DataFrame({
        deo_col:deo_grp.size().index,
        "total_projects":deo_grp.size().values,
        "green_flag_projects":deo_grp[tag_col].apply(lambda x:x.apply(lambda s:contains_category(s,"Green_Flag")).sum()),
        "total_contract_cost":deo_grp[cost_col].apply(lambda s:ensure_numeric(s).sum())
    })
    gm["green_flag_density"]=gm["green_flag_projects"]/gm["total_projects"]
    gm_link,label=_linkify_projects(gm,deo_col,"green_flag_projects","Green_Flag")
    st.data_editor(gm_link,use_container_width=True,hide_index=True,
        column_config={
            deo_col:st.column_config.TextColumn("DistrictEngineeringOffice"),
            "total_projects":st.column_config.NumberColumn("total_projects",format="%,d"),
            "green_flag_projects":st.column_config.LinkColumn("green_flag_projects",display_text="${"+label+"}"),
            "total_contract_cost":st.column_config.NumberColumn("total_contract_cost",format="%,.2f"),
            "green_flag_density":st.column_config.NumberColumn("green_flag_density",format=".3f"),
            label:None
        })
    st.subheader("Green vs Red (Top 20 DEO)")
    st.altair_chart(stacked_deo_share_chart(df,deo_col,tag_col,"Green_Flag",20),use_container_width=True)
    _drilldown_panel(df,deo_col,tag_col)

# ========== CATEGORY TABS ==========
def render_category(cat,container):
    with container:
        st.header(cat.replace("_"," "))
        deo_grp=df.groupby(deo_col,dropna=False)
        cm=pd.DataFrame({
            deo_col:deo_grp.size().index,
            "total_projects":deo_grp.size().values,
            "category_projects":deo_grp[tag_col].apply(lambda x:x.apply(lambda s:contains_category(s,cat)).sum())
        })
        cm_link,label=_linkify_projects(cm,deo_col,"category_projects",cat)
        st.data_editor(cm_link,use_container_width=True,hide_index=True,
            column_config={
                deo_col:st.column_config.TextColumn("DistrictEngineeringOffice"),
                "total_projects":st.column_config.NumberColumn("total_projects",format="%,d"),
                "category_projects":st.column_config.LinkColumn("category_projects",display_text="${"+label+"}"),
                label:None
            })
        st.subheader(f"{cat} vs Others (Top 20 DEO)")
        st.altair_chart(stacked_deo_share_chart(df,deo_col,tag_col,cat,20),use_container_width=True)
        _drilldown_panel(df,deo_col,tag_col)

render_category("Siyam-siyam_Project",tabs[2])
render_category("Chop_chop_Project",tabs[3])
render_category("Doppelganger_Project",tabs[4])
render_category("Ghost_Project",tabs[5])
