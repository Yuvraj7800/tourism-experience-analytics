"""
Tourism Experience Analytics — Streamlit App
Covers: Rating Prediction · Visit Mode Classification · Attraction Recommendation · EDA Dashboard
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, f1_score, classification_report)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.4rem 0;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a73e8;
        border-bottom: 2px solid #e8f0fe;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING  (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    transaction = pd.read_excel("Transaction.xlsx")
    user        = pd.read_excel("User.xlsx")
    item        = pd.read_excel("Item.xlsx")
    mode        = pd.read_excel("Mode.xlsx")
    atype       = pd.read_excel("Type.xlsx")
    city        = pd.read_excel("City.xlsx")
    country     = pd.read_excel("Country.xlsx")
    region      = pd.read_excel("Region.xlsx")
    continent   = pd.read_excel("Continent.xlsx")

    # ── clean ──────────────────────────────────────────────────────────────
    user["CityId"] = user["CityId"].fillna(0).astype(int)
    transaction    = transaction[transaction["Rating"].between(1, 5)].drop_duplicates().reset_index(drop=True)

    for df_ref in [city, country, region, continent, mode]:
        id_col = [c for c in df_ref.columns if c.endswith("Id")][0]
        df_ref.drop(df_ref[df_ref[id_col] == 0].index, inplace=True)

    # ── merge ──────────────────────────────────────────────────────────────
    df = transaction.copy()
    df = df.merge(mode.rename(columns={"VisitModeId": "VisitMode", "VisitMode": "VisitModeName"}),
                  on="VisitMode", how="left")
    df = df.merge(user, on="UserId", how="left")
    df = df.merge(continent.rename(columns={"Continent": "ContinentName"}), on="ContinentId", how="left")
    df = df.merge(region[["RegionId","Region"]].rename(columns={"Region":"RegionName"}), on="RegionId", how="left")
    df = df.merge(country[["CountryId","Country"]].rename(columns={"Country":"CountryName"}), on="CountryId", how="left")
    df = df.merge(city[["CityId","CityName"]].rename(columns={"CityName":"UserCityName"}), on="CityId", how="left")
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(atype, on="AttractionTypeId", how="left")
    df = df.merge(city[["CityId","CityName"]].rename(
                  columns={"CityId":"AttractionCityId","CityName":"AttractionCityName"}),
                  on="AttractionCityId", how="left")

    # ── feature engineering ───────────────────────────────────────────────
    def to_season(m):
        return {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                6:"Summer",7:"Summer",8:"Summer"}.get(m, "Autumn")
    df["Season"] = df["VisitMonth"].apply(to_season)
    df = df.join(df.groupby("AttractionId")["Rating"].mean().rename("AttractionAvgRating"), on="AttractionId")
    df = df.join(df.groupby("UserId")["Rating"].mean().rename("UserAvgRating"), on="UserId")
    df = df.join(df.groupby("UserId")["TransactionId"].count().rename("UserVisitCount"), on="UserId")
    df = df.join(df.groupby("AttractionId")["TransactionId"].count().rename("AttractionPopularity"), on="AttractionId")

    # ── encode ────────────────────────────────────────────────────────────
    le_dict = {}
    for col in ["VisitModeName","ContinentName","RegionName","CountryName","AttractionType","Season"]:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, item, atype, city, country, region, continent, mode, le_dict


@st.cache_resource(show_spinner=False)
def train_models(_df, _le_dict):
    feature_cols = ["VisitYear","VisitMonth","ContinentName_enc","RegionName_enc",
                    "AttractionType_enc","Season_enc","AttractionAvgRating","UserAvgRating",
                    "UserVisitCount","AttractionPopularity","AttractionTypeId"]

    ml = _df[feature_cols + ["Rating","VisitModeName_enc","VisitModeName"]].dropna()
    X  = ml[feature_cols]
    yr = ml["Rating"]
    yc = ml["VisitModeName_enc"]

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, yr, yc, test_size=0.2, random_state=42, stratify=yc)

    # Regression — XGBoost
    rgr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                            max_depth=6, random_state=42, verbosity=0)
    rgr.fit(X_tr, yr_tr)
    yr_pred = rgr.predict(X_te)
    reg_metrics = {
        "RMSE": round(float(np.sqrt(mean_squared_error(yr_te, yr_pred))), 4),
        "R2":   round(float(r2_score(yr_te, yr_pred)), 4),
    }

    # Classification — LightGBM
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                              num_leaves=63, random_state=42, verbose=-1)
    clf.fit(X_tr, yc_tr)
    yc_pred = clf.predict(X_te)
    clf_metrics = {
        "Accuracy": round(float(accuracy_score(yc_te, yc_pred)), 4),
        "F1":       round(float(f1_score(yc_te, yc_pred, average="weighted")), 4),
    }
    clf_report = classification_report(yc_te, yc_pred,
                                        target_names=_le_dict["VisitModeName"].classes_,
                                        output_dict=True)

    return rgr, clf, feature_cols, reg_metrics, clf_metrics, clf_report


@st.cache_resource(show_spinner=False)
def build_recommender(_df, _item, _atype):
    pivot = _df.pivot_table(index="UserId", columns="AttractionId",
                             values="Rating", aggfunc="mean")
    pivot_filled = pivot.apply(lambda r: r.fillna(r.mean()), axis=1)
    svd = TruncatedSVD(n_components=20, random_state=42)
    U   = svd.fit_transform(pivot_filled)
    Vt  = svd.components_
    pred_df = pd.DataFrame(np.dot(U, Vt), index=pivot.index, columns=pivot.columns)

    att_f = _item[["AttractionId","AttractionTypeId","AttractionCityId"]].copy()
    att_f = att_f.merge(_df.groupby("AttractionId")["Rating"].mean().rename("AvgRating"),
                         on="AttractionId", how="left").fillna(0)
    att_f = att_f.merge(_df.groupby("AttractionId")["TransactionId"].count().rename("Popularity"),
                         on="AttractionId", how="left").fillna(0)
    sc  = StandardScaler()
    mat = sc.fit_transform(att_f[["AttractionTypeId","AttractionCityId","AvgRating","Popularity"]])
    knn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
    knn.fit(mat)
    idx_map = {aid: i for i, aid in enumerate(att_f["AttractionId"])}

    return pred_df, pivot, knn, mat, idx_map, att_f


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
LOADED = False
LOAD_ERR = ""
with st.spinner("🔄 Loading data and training models — runs once then cached…"):
    try:
        df, item, atype, city, country, region, continent, mode_df, le_dict = load_data()
        rgr, clf, feature_cols, reg_metrics, clf_metrics, clf_report = train_models(df, le_dict)
        pred_df, pivot, knn, knn_mat, idx_map, att_f = build_recommender(df, item, atype)
        LOADED = True
    except Exception as e:
        LOAD_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Tourism Analytics")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠 Home Dashboard",
        "📊 EDA Explorer",
        "⭐ Predict Rating",
        "🧳 Predict Visit Mode",
        "🎯 Get Recommendations",
        "📈 Model Performance",
    ])
    st.markdown("---")
    if LOADED:
        st.success(f"✅ {len(df):,} transactions loaded")
        st.info(f"👤 {df['UserId'].nunique():,} users")
        st.info(f"🗺️ {df['AttractionId'].nunique()} attractions")
        st.info(f"⭐ Avg Rating: {df['Rating'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# ERROR STATE
# ─────────────────────────────────────────────────────────────────────────────
if not LOADED:
    st.error(f"❌ Could not load data: {LOAD_ERR}")
    st.info("👉 Make sure all `.xlsx` files are in the **same folder** as `app.py`, then restart.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class='main-header'>
        <h1 style='margin:0'>🌍 Tourism Experience Analytics</h1>
        <p style='margin:0.4rem 0 0 0; opacity:0.9'>
            Classification · Prediction · Recommendation System
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Users",       f"{df['UserId'].nunique():,}")
    c3.metric("Attractions",        f"{df['AttractionId'].nunique()}")
    c4.metric("Avg Rating",         f"{df['Rating'].mean():.2f} / 5")
    c5.metric("Visit Modes",        f"{df['VisitModeName'].nunique()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>📍 Users by Continent</div>", unsafe_allow_html=True)
        cont = df.drop_duplicates("UserId")["ContinentName"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        cont.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(cont)))
        ax.set_xlabel(""); ax.set_ylabel("Users")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>🧳 Visit Mode Share</div>", unsafe_allow_html=True)
        mc = df["VisitModeName"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(mc, labels=mc.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel", len(mc)))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>⭐ Rating Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        df["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax,
            color="steelblue", edgecolor="white")
        ax.set_xlabel("Rating"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("<div class='section-title'>📅 Monthly Visit Trend</div>", unsafe_allow_html=True)
        monthly = df.groupby("VisitMonth").size()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(monthly.index, monthly.values, marker="o", color="orangered", lw=2)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="orangered")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        ax.set_ylabel("Visits")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("<div class='section-title'>🏆 Top 10 Most Visited Attractions</div>", unsafe_allow_html=True)
    top10 = (df.groupby(["AttractionId","Attraction"])
               .agg(Visits=("TransactionId","count"), AvgRating=("Rating","mean"))
               .sort_values("Visits", ascending=False).head(10).reset_index())
    top10["AvgRating"] = top10["AvgRating"].round(2)
    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.barh(top10["Attraction"][::-1], top10["Visits"][::-1],
                   color=sns.color_palette("Blues_d", len(top10)))
    for bar, v in zip(bars, top10["Visits"][::-1]):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
                f'{v:,}', va='center', fontsize=9)
    ax.set_xlabel("Number of Visits"); ax.set_title("Top 10 Most Visited Attractions")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TOP REGIONS ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🗺️ Top 15 Regions by Visit Volume & Avg Rating</div>",
                unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        top_regions = (df.groupby("RegionName")
                         .agg(Visits=("TransactionId","count"))
                         .sort_values("Visits", ascending=False)
                         .head(15).reset_index())
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = sns.color_palette("YlOrRd", len(top_regions))[::-1]
        ax.barh(top_regions["RegionName"][::-1], top_regions["Visits"][::-1], color=colors)
        ax.set_title("Top 15 Regions — Visit Volume", fontweight="bold")
        ax.set_xlabel("Number of Visits")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r2:
        top_regions_rating = (df.groupby("RegionName")
                                .agg(AvgRating=("Rating","mean"), Visits=("TransactionId","count"))
                                .query("Visits >= 100")
                                .sort_values("AvgRating", ascending=False)
                                .head(15).reset_index())
        fig, ax = plt.subplots(figsize=(7, 5))
        colors2 = sns.color_palette("Greens", len(top_regions_rating))[::-1]
        bars2 = ax.barh(top_regions_rating["RegionName"][::-1],
                        top_regions_rating["AvgRating"][::-1], color=colors2)
        for bar, v in zip(bars2, top_regions_rating["AvgRating"][::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=9)
        ax.set_title("Top 15 Regions — Avg Rating (min 100 visits)", fontweight="bold")
        ax.set_xlabel("Average Rating"); ax.set_xlim(0, 5.5)
        ax.axvline(df["Rating"].mean(), color="red", linestyle="--", linewidth=1, label="Overall Mean")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── USER SEGMENTS ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>👥 User Segments Analysis</div>",
                unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        # Segment by visit mode preference
        user_mode = (df.groupby(["UserId","VisitModeName"])
                       .size().reset_index(name="Count")
                       .sort_values("Count", ascending=False)
                       .drop_duplicates("UserId"))
        seg_mode = user_mode["VisitModeName"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            seg_mode, labels=seg_mode.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2", len(seg_mode)),
            startangle=90, pctdistance=0.82)
        for t in autotexts: t.set_fontsize(9)
        ax.set_title("Users by Dominant\nVisit Mode", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_s2:
        # Segment by engagement (visit count buckets)
        user_visits = df.groupby("UserId")["TransactionId"].count()
        bins   = [0, 1, 3, 7, 15, 999]
        labels = ["1 visit","2–3 visits","4–7 visits","8–15 visits","16+ visits"]
        user_seg = pd.cut(user_visits, bins=bins, labels=labels)
        seg_counts = user_seg.value_counts().reindex(labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(seg_counts.index, seg_counts.values,
               color=sns.color_palette("Blues", len(seg_counts)))
        ax.set_title("Users by Engagement\n(Visit Count)", fontweight="bold")
        ax.set_ylabel("Number of Users")
        ax.tick_params(axis="x", rotation=20)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_s3:
        # Segment by rating behaviour (avg rating buckets)
        user_avg = df.groupby("UserId")["Rating"].mean()
        rating_bins   = [0, 2, 3, 4, 4.5, 5.1]
        rating_labels = ["≤2 (Critical)","2–3 (Low)","3–4 (Moderate)","4–4.5 (Happy)","4.5–5 (Delighted)"]
        user_rating_seg = pd.cut(user_avg, bins=rating_bins, labels=rating_labels)
        rseg_counts = user_rating_seg.value_counts().reindex(rating_labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        palette = ["#e53935","#fb8c00","#fdd835","#43a047","#1e88e5"]
        ax.barh(rseg_counts.index[::-1], rseg_counts.values[::-1], color=palette[::-1])
        ax.set_title("Users by Rating\nBehaviour", fontweight="bold")
        ax.set_xlabel("Number of Users")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Segment summary table
    st.markdown("**📊 Segment Summary**")
    seg_summary = (df.groupby("VisitModeName")
                     .agg(Users=("UserId","nunique"),
                          TotalVisits=("TransactionId","count"),
                          AvgRating=("Rating","mean"),
                          AvgVisitsPerUser=("TransactionId","count"))
                     .reset_index())
    seg_summary["AvgVisitsPerUser"] = (seg_summary["TotalVisits"] /
                                        seg_summary["Users"]).round(1)
    seg_summary["AvgRating"] = seg_summary["AvgRating"].round(2)
    seg_summary = seg_summary.sort_values("Users", ascending=False).reset_index(drop=True)
    st.dataframe(seg_summary[["VisitModeName","Users","TotalVisits",
                               "AvgRating","AvgVisitsPerUser"]].rename(
        columns={"VisitModeName":"Segment","AvgVisitsPerUser":"Avg Visits/User"}),
        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA Explorer":
    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📊 Exploratory Data Analysis</h2>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Geography", "⭐ Ratings", "🧳 Visit Modes", "🗓️ Time Trends"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            top_countries = df.drop_duplicates("UserId")["CountryName"].value_counts().head(12)
            top_countries.plot(kind="barh", ax=ax, color="teal")
            ax.set_title("Top 12 Countries by Users"); ax.set_xlabel("Users")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            cont_rating = df.groupby("ContinentName")["Rating"].mean().sort_values(ascending=False)
            cont_rating.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(cont_rating)))
            ax.set_title("Avg Rating by Continent"); ax.set_ylabel("Avg Rating")
            ax.set_ylim(0, 5.5); ax.tick_params(axis="x", rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            order = df.groupby("ContinentName")["Rating"].mean().sort_values(ascending=False).index
            sns.boxplot(data=df, x="ContinentName", y="Rating", order=order, palette="Set2", ax=ax)
            ax.set_title("Rating by Continent"); ax.tick_params(axis="x", rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            type_rating = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False).head(12)
            type_rating.plot(kind="barh", ax=ax, color="coral")
            ax.set_title("Avg Rating by Attraction Type (Top 12)")
            ax.axvline(df["Rating"].mean(), color="navy", linestyle="--", label="Overall Mean")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

        num_cols = ["Rating","VisitYear","VisitMonth","AttractionAvgRating",
                    "UserAvgRating","UserVisitCount","AttractionPopularity"]
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            mode_rating = df.groupby("VisitModeName")["Rating"].mean().sort_values(ascending=False)
            mode_rating.plot(kind="bar", ax=ax, color=sns.color_palette("Set3", len(mode_rating)))
            ax.set_title("Avg Rating by Visit Mode"); ax.set_ylabel("Avg Rating")
            ax.set_ylim(0, 5.5); ax.axhline(df["Rating"].mean(), color="red", linestyle="--")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            season_order = [s for s in ["Spring","Summer","Autumn","Winter"]
                            if s in df["Season"].unique()]
            fig, ax = plt.subplots(figsize=(7, 4))
            smode = df.groupby(["Season","VisitModeName"]).size().unstack(fill_value=0)
            smode = smode.reindex([s for s in season_order if s in smode.index])
            smode.plot(kind="bar", ax=ax, stacked=True, colormap="Set2", edgecolor="white")
            ax.set_title("Visit Mode by Season"); ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=20); ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            yearly = df.groupby("VisitYear").size()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(yearly.index.astype(str), yearly.values, color="steelblue", edgecolor="white")
            ax.set_title("Transactions per Year"); ax.set_ylabel("Count")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            monthly2 = df.groupby("VisitMonth").agg(
                Visits=("TransactionId","count"), AvgRating=("Rating","mean")).reset_index()
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax2 = ax1.twinx()
            ax1.bar(monthly2["VisitMonth"], monthly2["Visits"], color="steelblue", alpha=0.6)
            ax2.plot(monthly2["VisitMonth"], monthly2["AvgRating"],
                     color="orangered", marker="o", lw=2)
            ax1.set_xticks(range(1, 13))
            ax1.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
            ax1.set_ylabel("Visits", color="steelblue")
            ax2.set_ylabel("Avg Rating", color="orangered")
            ax1.set_title("Monthly Visits & Avg Rating")
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PREDICT RATING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⭐ Predict Rating":
    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>⭐ Predict Attraction Rating</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.9'>XGBoost Regression Model</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 User Profile")
        continent_opts = sorted(df["ContinentName"].dropna().unique())
        sel_continent  = st.selectbox("Continent", continent_opts)
        region_opts = sorted(df[df["ContinentName"]==sel_continent]["RegionName"].dropna().unique())
        sel_region  = st.selectbox("Region", region_opts if region_opts else
                                   sorted(df["RegionName"].dropna().unique()))
        user_avg_rating  = st.slider("User's Historical Avg Rating", 1.0, 5.0, 3.5, 0.1)
        user_visit_count = st.number_input("User's Total Past Visits", 1, 500, 10)

    with col2:
        st.markdown("#### 🗺️ Visit Details")
        visit_year  = st.selectbox("Visit Year", sorted(df["VisitYear"].unique(), reverse=True))
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        visit_month = st.selectbox("Visit Month",
            [(i+1, m) for i, m in enumerate(month_names)],
            format_func=lambda x: x[1])
        visit_month_val = visit_month[0]
        season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer"}
        auto_season = season_map.get(visit_month_val, "Autumn")
        st.info(f"🌤️ Detected Season: **{auto_season}**")

    with col3:
        st.markdown("#### 🏛️ Attraction Details")
        type_opts = sorted(df["AttractionType"].dropna().unique())
        sel_type  = st.selectbox("Attraction Type", type_opts)
        type_id   = int(atype[atype["AttractionType"]==sel_type]["AttractionTypeId"].values[0])
        att_avg   = float(df[df["AttractionType"]==sel_type]["Rating"].mean())
        att_pop   = int(df[df["AttractionType"]==sel_type]["TransactionId"].count())
        att_avg_override = st.slider("Attraction Avg Rating", 1.0, 5.0, round(att_avg, 1), 0.1)
        att_pop_override = st.number_input("Attraction Popularity (visits)", 1, 50000, att_pop)

    st.markdown("---")
    if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
        cont_enc   = le_dict["ContinentName"].transform([sel_continent])[0]
        region_enc = (le_dict["RegionName"].transform([sel_region])[0]
                      if sel_region in le_dict["RegionName"].classes_ else 0)
        type_enc   = le_dict["AttractionType"].transform([sel_type])[0]
        season_enc = le_dict["Season"].transform([auto_season])[0]

        X_input = pd.DataFrame([[
            visit_year, visit_month_val,
            cont_enc, region_enc, type_enc, season_enc,
            att_avg_override, user_avg_rating,
            user_visit_count, att_pop_override, type_id
        ]], columns=feature_cols)

        predicted = float(rgr.predict(X_input)[0])
        predicted = max(1.0, min(5.0, predicted))
        stars = "⭐" * round(predicted)

        _, col_b, _ = st.columns([1, 2, 1])
        with col_b:
            st.markdown(f"""
            <div style='text-align:center; background:#e3f2fd; padding:2rem;
                        border-radius:16px; border:2px solid #1a73e8'>
                <div style='font-size:3.5rem'>{stars}</div>
                <div style='font-size:2.8rem; font-weight:800; color:#1a73e8'>
                    {predicted:.2f} / 5.0
                </div>
                <div style='color:#555; margin-top:0.5rem'>Predicted Rating</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Model Test-Set Performance**")
        m1, m2 = st.columns(2)
        m1.metric("RMSE", reg_metrics["RMSE"])
        m2.metric("R²",   reg_metrics["R2"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PREDICT VISIT MODE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧳 Predict Visit Mode":
    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🧳 Predict Visit Mode</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.9'>LightGBM Classification Model</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 User Profile")
        continent_opts = sorted(df["ContinentName"].dropna().unique())
        sel_continent  = st.selectbox("Continent", continent_opts, key="clf_cont")
        region_opts    = sorted(df[df["ContinentName"]==sel_continent]["RegionName"].dropna().unique())
        sel_region     = st.selectbox("Region",
                                      region_opts if region_opts else sorted(df["RegionName"].dropna().unique()),
                                      key="clf_reg")
        user_avg_rating  = st.slider("User's Avg Historical Rating", 1.0, 5.0, 3.5, 0.1, key="clf_uar")
        user_visit_count = st.number_input("User's Total Past Visits", 1, 500, 15, key="clf_uvc")

    with col2:
        st.markdown("#### 🗺️ Visit Details")
        visit_year  = st.selectbox("Visit Year", sorted(df["VisitYear"].unique(), reverse=True), key="clf_yr")
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        visit_month = st.selectbox("Visit Month",
            [(i+1, m) for i, m in enumerate(month_names)],
            format_func=lambda x: x[1], key="clf_mo")
        visit_month_val = visit_month[0]
        season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer"}
        auto_season = season_map.get(visit_month_val, "Autumn")
        st.info(f"🌤️ Season: **{auto_season}**")

    with col3:
        st.markdown("#### 🏛️ Attraction Details")
        type_opts = sorted(df["AttractionType"].dropna().unique())
        sel_type  = st.selectbox("Attraction Type", type_opts, key="clf_type")
        type_id   = int(atype[atype["AttractionType"]==sel_type]["AttractionTypeId"].values[0])
        att_avg   = float(df[df["AttractionType"]==sel_type]["Rating"].mean())
        att_pop   = int(df[df["AttractionType"]==sel_type]["TransactionId"].count())
        att_avg_override = st.slider("Attraction Avg Rating", 1.0, 5.0, round(att_avg, 1), 0.1, key="clf_aar")
        att_pop_override = st.number_input("Attraction Popularity", 1, 50000, att_pop, key="clf_pop")

    st.markdown("---")
    if st.button("🔮 Predict Visit Mode", type="primary", use_container_width=True):
        cont_enc   = le_dict["ContinentName"].transform([sel_continent])[0]
        region_enc = (le_dict["RegionName"].transform([sel_region])[0]
                      if sel_region in le_dict["RegionName"].classes_ else 0)
        type_enc   = le_dict["AttractionType"].transform([sel_type])[0]
        season_enc = le_dict["Season"].transform([auto_season])[0]

        X_input = pd.DataFrame([[
            visit_year, visit_month_val,
            cont_enc, region_enc, type_enc, season_enc,
            att_avg_override, user_avg_rating,
            user_visit_count, att_pop_override, type_id
        ]], columns=feature_cols)

        pred_enc   = clf.predict(X_input)[0]
        pred_proba = clf.predict_proba(X_input)[0]
        pred_mode  = le_dict["VisitModeName"].inverse_transform([pred_enc])[0]
        classes    = le_dict["VisitModeName"].classes_

        mode_icons = {"Business":"💼","Couples":"💑","Family":"👨‍👩‍👧","Friends":"👫","Solo":"🧍"}
        icon = mode_icons.get(pred_mode, "🧳")

        _, col_b, _ = st.columns([1, 2, 1])
        with col_b:
            st.markdown(f"""
            <div style='text-align:center; background:#e8f5e9; padding:2rem;
                        border-radius:16px; border:2px solid #4caf50'>
                <div style='font-size:3.5rem'>{icon}</div>
                <div style='font-size:2.4rem; font-weight:800; color:#2e7d32'>
                    {pred_mode}
                </div>
                <div style='color:#555; margin-top:0.5rem'>Predicted Visit Mode</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Confidence per Visit Mode")
        prob_df = pd.DataFrame({"Mode": classes, "Probability": pred_proba}).sort_values("Probability")
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["#4caf50" if m == pred_mode else "steelblue" for m in prob_df["Mode"]]
        ax.barh(prob_df["Mode"], prob_df["Probability"], color=colors)
        ax.set_xlabel("Probability"); ax.set_xlim(0, 1)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", clf_metrics["Accuracy"])
        m2.metric("F1-Score", clf_metrics["F1"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Get Recommendations":
    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🎯 Personalized Attraction Recommendations</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.9'>Collaborative · Content-Based · Hybrid</p>
    </div>""", unsafe_allow_html=True)

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🤝 Collaborative Filtering",
        "🔍 Content-Based Filtering",
        "⚡ Hybrid Recommendation"
    ])

    with rec_tab1:
        st.markdown("Recommends attractions based on **similar users' ratings** (SVD Matrix Factorization).")
        top_users = df["UserId"].value_counts().head(300).index.tolist()
        sel_user  = st.selectbox("Select User ID", top_users, key="cf_user")
        top_n_cf  = st.slider("Number of Recommendations", 5, 20, 10, key="cf_n")

        if st.button("🎯 Get Recommendations", type="primary", key="cf_btn"):
            if sel_user in pred_df.index:
                visited = pivot.loc[sel_user].dropna().index.tolist()
                scores  = (pred_df.loc[sel_user]
                             .drop(index=visited, errors="ignore")
                             .sort_values(ascending=False)
                             .head(top_n_cf))
                result  = pd.DataFrame({"AttractionId": scores.index, "Predicted Rating": scores.values.round(2)})
                result  = result.merge(item[["AttractionId","Attraction","AttractionTypeId"]], on="AttractionId", how="left")
                result  = result.merge(atype, on="AttractionTypeId", how="left")
                st.dataframe(result[["Attraction","AttractionType","Predicted Rating"]].reset_index(drop=True),
                             use_container_width=True)
                st.caption(f"User {sel_user} has visited {len(visited)} attractions. Showing {top_n_cf} new suggestions.")
            else:
                st.warning("User not found in training data.")

    with rec_tab2:
        st.markdown("Finds attractions **similar to a chosen one**, based on type, location & popularity.")
        att_list    = item[["AttractionId","Attraction"]].drop_duplicates()
        att_options = dict(zip(att_list["Attraction"], att_list["AttractionId"]))
        sel_att_name = st.selectbox("Select an Attraction", sorted(att_options.keys()), key="cb_att")
        top_n_cb     = st.slider("Number of Similar Attractions", 5, 20, 10, key="cb_n")

        if st.button("🔍 Find Similar Attractions", type="primary", key="cb_btn"):
            sel_att_id = att_options[sel_att_name]
            if sel_att_id in idx_map:
                idx = idx_map[sel_att_id]
                dists, idxs = knn.kneighbors([knn_mat[idx]])
                sim_ids = att_f.iloc[idxs[0][1:top_n_cb+1]]["AttractionId"].values
                result  = item[item["AttractionId"].isin(sim_ids)][["AttractionId","Attraction","AttractionTypeId"]].copy()
                result  = result.merge(atype, on="AttractionTypeId", how="left")
                result  = result.merge(
                    df.groupby("AttractionId")["Rating"].mean().rename("Avg Rating").round(2),
                    on="AttractionId", how="left")
                st.dataframe(result[["Attraction","AttractionType","Avg Rating"]].reset_index(drop=True),
                             use_container_width=True)
            else:
                st.warning("Attraction not found.")

    with rec_tab3:
        st.markdown("Combines **collaborative + content-based** signals for maximum accuracy.")
        top_users_h  = df["UserId"].value_counts().head(300).index.tolist()
        sel_user_h   = st.selectbox("Select User ID", top_users_h, key="hy_user")
        alpha_val    = st.slider("Collaborative Weight (α)",  0.1, 0.9, 0.6, 0.1,
                                 help="α=1 → purely collaborative | α=0 → purely content-based")
        top_n_h      = st.slider("Number of Recommendations", 5, 20, 10, key="hy_n")

        if st.button("⚡ Get Hybrid Recommendations", type="primary", key="hy_btn"):
            if sel_user_h in pred_df.index:
                visited      = pivot.loc[sel_user_h].dropna().index.tolist()
                collab       = pred_df.loc[sel_user_h].drop(index=visited, errors="ignore")
                user_visited = df[df["UserId"]==sel_user_h].nlargest(3,"Rating")["AttractionId"].tolist()

                content_scores = {}
                for av in user_visited:
                    if av in idx_map:
                        idx = idx_map[av]
                        dists, idxs = knn.kneighbors([knn_mat[idx]])
                        for d, i in zip(dists[0][1:], idxs[0][1:]):
                            aid = att_f.iloc[i]["AttractionId"]
                            content_scores[aid] = content_scores.get(aid, 0) + (1 - d)

                cs = pd.Series(content_scores)
                cs = cs / cs.max() if cs.max() > 0 else cs
                cn = (collab - collab.min()) / (collab.max() - collab.min() + 1e-6)
                combined = (alpha_val * cn + (1 - alpha_val) * cs).sort_values(ascending=False).head(top_n_h)

                result = pd.DataFrame({"AttractionId": combined.index, "Hybrid Score": combined.values.round(4)})
                result = result.merge(item[["AttractionId","Attraction","AttractionTypeId"]], on="AttractionId", how="left")
                result = result.merge(atype, on="AttractionTypeId", how="left")
                result = result.merge(
                    df.groupby("AttractionId")["Rating"].mean().rename("Avg Rating").round(2),
                    on="AttractionId", how="left")
                st.dataframe(result[["Attraction","AttractionType","Avg Rating","Hybrid Score"]].reset_index(drop=True),
                             use_container_width=True)
                st.caption(f"α={alpha_val} → {int(alpha_val*100)}% collaborative + {int((1-alpha_val)*100)}% content-based")
            else:
                st.warning("User not found.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📈 Model Performance</h2>
    </div>""", unsafe_allow_html=True)

    tab_r, tab_c = st.tabs(["📉 Regression — Rating Prediction",
                             "🎯 Classification — Visit Mode"])

    with tab_r:
        c1, c2 = st.columns(2)
        c1.metric("RMSE (XGBoost)", reg_metrics["RMSE"],
                  help="Lower is better. Root Mean Squared Error on test set.")
        c2.metric("R² Score", reg_metrics["R2"],
                  help="Closer to 1.0 is better.")
        st.markdown("---")
        st.markdown("#### Feature Importance — XGBoost Regressor")
        fi = pd.Series(rgr.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        fi.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_ylabel("Importance"); ax.tick_params(axis="x", rotation=35, labelsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab_c:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy (LightGBM)", clf_metrics["Accuracy"])
        c2.metric("F1-Score (weighted)", clf_metrics["F1"])
        st.markdown("---")
        st.markdown("#### Per-Class Classification Report")
        report_df = (pd.DataFrame(clf_report).T
                       .drop(["accuracy","macro avg","weighted avg"], errors="ignore")
                       .rename(columns={"precision":"Precision","recall":"Recall",
                                        "f1-score":"F1","support":"Support"})
                       [["Precision","Recall","F1","Support"]]
                       .round(3))
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Feature Importance — LightGBM Classifier")
        fi_c = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        fi_c.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")
        ax.set_ylabel("Importance"); ax.tick_params(axis="x", rotation=35, labelsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
