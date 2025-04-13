# dashboard.py

"""
Streamlit dashboard to visualize recommendation system metrics.
Includes model performance, engagement, and bandit vs static comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# === Load Data ===
def load_metrics(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        return pd.read_json(path)
    else:
        raise ValueError("Unsupported metrics format")


def load_logs(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# === Streamlit Layout ===
st.title("📊 Recommendation System Dashboard")
st.sidebar.header("Configuration")

metrics_path = st.sidebar.file_uploader("Upload evaluation metrics (CSV/JSON)", type=["csv", "json"])
log_file = st.sidebar.file_uploader("Upload user interaction log", type="csv")

if metrics_path:
    metrics_df = load_metrics(Path(metrics_path.name))
    st.subheader("📈 Model Performance Metrics")
    st.write(metrics_df)

if log_file:
    log_df = load_logs(Path(log_file.name))
    st.subheader("🧭 Engagement Trends")

    with st.expander("Clicks and Conversions Over Time"):
        daily = log_df.groupby(log_df["timestamp"].dt.date).agg({
            "click": "sum",
            "conversion": "sum"
        }).reset_index()
        fig = px.line(daily, x="timestamp", y=["click", "conversion"], title="User Engagement")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎯 Bandit vs Static Performance"):
        if "strategy" in log_df.columns:
            bandit_perf = log_df.groupby(["strategy", log_df["timestamp"].dt.date]).agg({
                "click": "sum",
                "conversion": "sum"
            }).reset_index()
            fig = px.line(
                bandit_perf,
                x="timestamp",
                y="conversion",
                color="strategy",
                title="Conversions by Strategy Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy column found. Upload bandit log to enable comparison.")