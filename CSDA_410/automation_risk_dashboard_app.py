from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Automation Risk Dashboard", layout="wide")

st.title("Occupation Automation Risk Dashboard")
st.markdown(
    """
    This dashboard analyzes occupation-level automation risk and employment distribution
    across U.S. states. It includes filtering, feature engineering, PCA, and K-means
    clustering to identify labor market patterns related to automation susceptibility.
    """
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


def make_risk_category(series: pd.Series) -> pd.Series:
    q1 = series.quantile(1 / 3)
    q2 = series.quantile(2 / 3)

    return pd.cut(
        series,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"]
    )


def build_cluster_summary(df_with_clusters, occupation_col, prob_col):
    summary = (
        df_with_clusters.groupby("Cluster")
        .agg(
            Occupation_Count=(occupation_col, "count"),
            Mean_Automation_Risk=(prob_col, "mean"),
            Mean_Total_Employment=("total_employment_raw", "mean"),
            Mean_States_Present=("states_present", "mean"),
        )
        .reset_index()
    )

    examples = (
        df_with_clusters.groupby("Cluster")[occupation_col]
        .first()
        .reset_index()
        .rename(columns={occupation_col: "Example Occupation"})
    )

    summary = summary.merge(examples, on="Cluster", how="left")
    return summary


try:
    df = load_data()
    st.success(f"Loaded file: {DATA_FILE.name}")
except Exception as e:
    st.error(f"Could not load file: {DATA_FILE}")
    st.exception(e)
    st.stop()

# -----------------------------
# Explicit dataset structure
# -----------------------------
prob_col = "Probability"
occupation_col = "Occupation"
id_col = "SOC"

state_cols = [
    c for c in df.columns
    if c not in [id_col, occupation_col, prob_col]
]

st.sidebar.header("Filters")

min_prob = float(df[prob_col].min())
max_prob = float(df[prob_col].max())

prob_range = st.sidebar.slider(
    "Filter automation probability",
    min_value=min_prob,
    max_value=max_prob,
    value=(min_prob, max_prob),
)

filtered_df = df[
    (df[prob_col] >= prob_range[0]) &
    (df[prob_col] <= prob_range[1])
].copy()

search_term = st.sidebar.text_input("Search occupation")
if search_term:
    filtered_df = filtered_df[
        filtered_df[occupation_col].astype(str).str.contains(
            search_term, case=False, na=False
        )
    ]

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Occupations", len(filtered_df))
col2.metric("Mean Automation Risk", f"{filtered_df[prob_col].mean():.3f}")
col3.metric("Median Automation Risk", f"{filtered_df[prob_col].median():.3f}")
col4.metric("Std. Dev. Risk", f"{filtered_df[prob_col].std():.3f}")

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

st.markdown(
    """
    **Interpretation:** This histogram shows how automation risk is distributed across
    occupations. If the shape appears bimodal, that suggests occupations are concentrated
    into lower-risk and higher-risk groups rather than being evenly spread across the range.
    """
)

# -----------------------------
# Highest Risk Occupations
# -----------------------------
st.subheader("Highest-Risk Occupations")
top_n = st.slider("Number of occupations to show", 5, 25, 10)
top_df = (
    filtered_df[[id_col, occupation_col, prob_col]]
    .sort_values(prob_col, ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
st.dataframe(top_df, use_container_width=True)

# -----------------------------
# Feature Engineering
# -----------------------------
working_df = filtered_df.copy()

working_df["total_employment_raw"] = working_df[state_cols].sum(axis=1)

for col in state_cols:
    working_df[col] = np.log1p(working_df[col])

working_df["total_employment"] = working_df[state_cols].sum(axis=1)
working_df["states_present"] = (working_df[state_cols] > 0).sum(axis=1)
working_df["risk_category"] = make_risk_category(working_df[prob_col])

st.subheader("Feature Engineering Summary")
st.write("The dashboard creates the following derived features:")
st.write("- **total_employment_raw**: sum of original employment counts across all states")
st.write("- **total_employment**: sum of log-transformed state employment values")
st.write("- **states_present**: number of states where the occupation appears")
st.write("- **risk_category**: Low / Medium / High automation risk using quantile-based cut-points")

q1 = working_df[prob_col].quantile(1 / 3)
q2 = working_df[prob_col].quantile(2 / 3)
st.info(
    f"Risk category cut-points: Low ≤ {q1:.3f}, Medium ≤ {q2:.3f}, High > {q2:.3f}"
)

# -----------------------------
# Employment vs Automation
# -----------------------------
st.subheader("Automation Risk vs Total Employment")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(working_df["total_employment_raw"], working_df[prob_col])
ax2.set_xlabel("Total Employment (Raw Sum Across States)")
ax2.set_ylabel("Automation Probability")
ax2.set_title("Automation Risk vs Total Employment")
st.pyplot(fig2)

st.markdown(
    """
    **Interpretation:** This plot helps identify whether occupations with larger employment
    footprints tend to have higher or lower automation risk, and whether the distribution
    suggests clustering or polarization across the labor market.
    """
)

# -----------------------------
# Risk Category Counts
# -----------------------------
st.subheader("Risk Category Counts")
risk_counts = (
    working_df["risk_category"]
    .value_counts()
    .rename_axis("Risk Category")
    .reset_index(name="Count")
)
st.dataframe(risk_counts, use_container_width=True)

# -----------------------------
# PCA
# -----------------------------
analysis_features = [prob_col] + state_cols + ["total_employment", "states_present"]
X = working_df[analysis_features].dropna()
aligned_df = working_df.loc[X.index].copy()

if len(X) >= 3:
    st.subheader("Standardization and PCA")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_components = min(len(analysis_features), len(X))
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_scaled)

    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    pca_components = st.slider(
        "Number of PCA components to retain",
        min_value=2,
        max_value=min(10, max_components),
        value=2
    )

    retained_variance = cumulative_variance[pca_components - 1]

    st.write(
        f"Data were standardized using **StandardScaler**. "
        f"The first **{pca_components}** principal components explain "
        f"**{retained_variance:.2%}** of the total variance."
    )

    scree_df = pd.DataFrame({
        "Component": np.arange(1, len(explained_variance) + 1),
        "Explained Variance Ratio": explained_variance,
        "Cumulative Variance": cumulative_variance
    })
    st.dataframe(scree_df.head(10), use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(scree_df["Component"], scree_df["Cumulative Variance"], marker="o")
    ax3.set_xlabel("Number of Components")
    ax3.set_ylabel("Cumulative Explained Variance")
    ax3.set_title("PCA Cumulative Explained Variance")
    st.pyplot(fig3)

    pca = PCA(n_components=pca_components)
    comps = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": comps[:, 0],
        "PC2": comps[:, 1],
        "Automation Probability": aligned_df[prob_col].values,
        "Occupation": aligned_df[occupation_col].values,
        "SOC": aligned_df[id_col].values,
        "risk_category": aligned_df["risk_category"].values,
        "total_employment_raw": aligned_df["total_employment_raw"].values,
        "states_present": aligned_df["states_present"].values
    })

    st.subheader("PCA Projection of Occupations")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    scatter = ax4.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["Automation Probability"]
    )
    ax4.set_xlabel("Principal Component 1")
    ax4.set_ylabel("Principal Component 2")
    ax4.set_title("2D PCA Projection")
    plt.colorbar(scatter, ax=ax4, label="Automation Probability")
    st.pyplot(fig4)

    # -----------------------------
    # K-Means
    # -----------------------------
    st.subheader("K-Means Clustering of Occupations")
    k = st.slider("Number of clusters", 2, 8, 3)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(comps)

    plot_df["Cluster"] = clusters
    aligned_df["Cluster"] = clusters

    sil_score = silhouette_score(comps, clusters) if len(np.unique(clusters)) > 1 else np.nan
    st.metric("Silhouette Score", f"{sil_score:.3f}")

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    for cluster in sorted(plot_df["Cluster"].unique()):
        cluster_data = plot_df[plot_df["Cluster"] == cluster]
        ax5.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}")

    ax5.set_xlabel("Principal Component 1")
    ax5.set_ylabel("Principal Component 2")
    ax5.set_title("Clusters in PCA Space")
    ax5.legend()
    st.pyplot(fig5)

    # -----------------------------
    # Cluster summary
    # -----------------------------
    st.subheader("Cluster Summary")
    cluster_summary = build_cluster_summary(
        aligned_df,
        occupation_col=occupation_col,
        prob_col=prob_col
    )
    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Representative Occupations by Cluster")
    for cluster_id in sorted(aligned_df["Cluster"].unique()):
        st.markdown(f"**Cluster {cluster_id}**")
        cluster_rows = aligned_df[aligned_df["Cluster"] == cluster_id][
            [id_col, occupation_col, prob_col, "total_employment_raw", "states_present", "risk_category"]
        ].sort_values(prob_col, ascending=False).head(5)
        st.dataframe(cluster_rows, use_container_width=True)

# -----------------------------
# Project Notes
# -----------------------------
st.subheader("Project Notes")
st.write(
    """
    This dashboard supports the final project report by combining preprocessing,
    quantile-based risk categorization, StandardScaler normalization, PCA dimensionality
    reduction, and K-means clustering in one interactive application. The added cluster
    summary tables and PCA variance reporting make the results easier to interpret and
    easier to document in the final paper.
    """
)
