import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Venue Insight Clustering", page_icon="🏟️")
st.title("🏟️ Venue Behavior Clustering")
st.markdown("See which venues are **spin-friendly**, **batting-heavy**, or **balanced**.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("venue_clustered_stats.csv")

df = load_data()

# Cluster labels
cluster_names = {
    0: "Spin-Friendly",
    1: "Batting Paradise",
    2: "Balanced/Seam-Assisting"
}
df['cluster_label'] = df['cluster'].map(cluster_names)

# Select view
selected_cluster = st.selectbox("Select Venue Type", df['cluster_label'].unique())
filtered = df[df['cluster_label'] == selected_cluster]

st.subheader(f"📍 Venues classified as: {selected_cluster}")
st.dataframe(filtered[['venue', 'avg_score', 'run_rate', 'is_spin', 'is_pace']])

# Plot heatmap
st.subheader("📊 Feature Heatmap by Cluster")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.groupby('cluster')[['avg_score', 'run_rate', 'is_spin', 'is_pace']].mean(),
            annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
st.pyplot(fig)
