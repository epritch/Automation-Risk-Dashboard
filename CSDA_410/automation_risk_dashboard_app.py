from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Automation Risk Dashboard", layout="wide")

st.title("Occupation Automation Risk Dashboard")
st.markdown(
    "This dashboard explores the relationship between occupation-level automation risk "
    "and employment distribution across U.S. states."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "automation_data_by_state.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_data()
    st.success(f"Loaded file: {DATA_FILE.name}")
except Exception as e:
    st.error(f"Could not load file: {DATA_FILE}")
    st.exception(e)
    st.stop()

# Try to infer important columns automatically
prob_candidates = [c for c in df.columns if "prob" in c.lower() or "automation" in c.lower()]
occ_candidates = [c for c in df.columns if "occupation" in c.lower() or "title" in c.lower()]

prob_col = st.sidebar.selectbox(
    "Automation probability column",
    prob_candidates if prob_candidates else df.columns,
)

occupation_col = st.sidebar.selectbox(
    "Occupation title column",
    occ_candidates if occ_candidates else df.columns,
)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
state_cols = [c for c in numeric_cols if c != prob_col]

# -----------------------------
# Filtering
# -----------------------------

min_prob = float(df[prob_col].min())
max_prob = float(df[prob_col].max())

prob_range = st.sidebar.slider(
    "Filter automation probability",
    min_value=min_prob,
    max_value=max_prob,
    value=(min_prob, max_prob),
)

filtered_df = df[(df[prob_col] >= prob_range[0]) & (df[prob_col] <= prob_range[1])].copy()

if occupation_col in filtered_df.columns:
    search_term = st.sidebar.text_input("Search occupation")
    if search_term:
        filtered_df = filtered_df[
            filtered_df[occupation_col].astype(str).str.contains(search_term, case=False, na=False)
        ]

# -----------------------------
# Dataset Preview
# -----------------------------

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Occupations", len(filtered_df))
col2.metric("Mean Automation Risk", f"{filtered_df[prob_col].mean():.3f}")
col3.metric("Median Automation Risk", f"{filtered_df[prob_col].median():.3f}")

# -----------------------------
# Histogram
# -----------------------------

st.subheader("Automation Risk Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.hist(filtered_df[prob_col].dropna(), bins=20)
ax1.set_xlabel("Automation Probability")
ax1.set_ylabel("Count")
ax1.set_title("Histogram of Automation Risk")
st.pyplot(fig1)

# -----------------------------
# Highest Risk Occupations
# -----------------------------

if occupation_col in filtered_df.columns:
    st.subheader("Highest-Risk Occupations")
    top_n = st.slider("Number of occupations to show", 5, 25, 10)
    top_df = filtered_df[[occupation_col, prob_col]].sort_values(prob_col, ascending=False).head(top_n)
    st.dataframe(top_df, use_container_width=True)

# -----------------------------
# Feature Engineering
# -----------------------------

working_df = filtered_df.copy()

if state_cols:
    for col in state_cols:
        working_df[col] = np.log1p(working_df[col])

    working_df["total_employment"] = working_df[state_cols].sum(axis=1)
    working_df["states_present"] = (working_df[state_cols] > 0).sum(axis=1)

    # -----------------------------
    # Employment vs Automation
    # -----------------------------

    st.subheader("Automation Risk vs Total Employment")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(working_df["total_employment"], working_df[prob_col])
    ax2.set_xlabel("Log Transformed Total Employment")
    ax2.set_ylabel("Automation Probability")
    st.pyplot(fig2)

    # -----------------------------
    # PCA
    # -----------------------------

    analysis_features = [prob_col] + state_cols + ["total_employment", "states_present"]
    analysis_features = [c for c in analysis_features if c in working_df.columns]

    X = working_df[analysis_features].dropna()

    if len(X) >= 3:
        st.subheader("PCA Projection of Occupations")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": comps[:, 0],
            "PC2": comps[:, 1],
            "Automation Probability": working_df.loc[X.index, prob_col].values
        })

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        scatter = ax3.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["Automation Probability"])
        ax3.set_xlabel("Principal Component 1")
        ax3.set_ylabel("Principal Component 2")
        plt.colorbar(scatter, ax=ax3, label="Automation Probability")
        st.pyplot(fig3)

        # -----------------------------
        # K-Means Clustering
        # -----------------------------

        st.subheader("K-Means Clustering of Occupations")
        k = st.slider("Number of clusters", 2, 8, 3)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_scaled)

        pca_df["Cluster"] = clusters.astype(str)

        fig4, ax4 = plt.subplots(figsize=(8, 5))

        for cluster in sorted(pca_df["Cluster"].unique()):
            cluster_data = pca_df[pca_df["Cluster"] == cluster]
            ax4.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}")

        ax4.set_xlabel("Principal Component 1")
        ax4.set_ylabel("Principal Component 2")
        ax4.legend()
        st.pyplot(fig4)

else:
    st.warning("No numeric state employment columns were detected.")

# -----------------------------
# Project Notes
# -----------------------------

st.subheader("Project Notes")
st.write(
    "This dashboard automatically loads the automation dataset from the local project folder. "
    "It includes filtering, risk distribution visualization, PCA dimensionality reduction, "
    "and clustering analysis to explore occupational automation risk patterns."
)
