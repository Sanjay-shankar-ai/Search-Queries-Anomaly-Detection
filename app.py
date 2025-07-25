# streamlit_app.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Search Query Anomaly Detector", layout="wide")
st.title("🔍 Search Queries Anomaly Detection")

# File upload
uploaded_file = st.file_uploader("Upload your Queries CSV file", type=["csv"])

if uploaded_file:
    queries_df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(queries_df.head())

    # Cleaning CTR column
    queries_df['CTR'] = queries_df['CTR'].str.rstrip('%').astype('float') / 100

    # Sidebar filters
    st.sidebar.header("Feature Visualizations")

    # Most common words in queries
    def clean_and_split(query):
        return re.findall(r'\b[a-zA-Z]+\b', query.lower())

    word_counts = Counter()
    for query in queries_df['Top queries']:
        word_counts.update(clean_and_split(query))
    word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

    st.subheader("📊 Top 20 Most Common Words in Queries")
    st.plotly_chart(px.bar(word_freq_df, x='Word', y='Frequency'))

    # Top by clicks and impressions
    st.subheader("Top Queries by Clicks and Impressions")
    col1, col2 = st.columns(2)
    with col1:
        top_clicks = queries_df.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
        st.plotly_chart(px.bar(top_clicks, x='Top queries', y='Clicks'))
    with col2:
        top_impressions = queries_df.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]
        st.plotly_chart(px.bar(top_impressions, x='Top queries', y='Impressions'))

    # Top and bottom CTR
    st.subheader("CTR Extremes")
    col3, col4 = st.columns(2)
    with col3:
        top_ctr = queries_df.nlargest(10, 'CTR')[['Top queries', 'CTR']]
        st.plotly_chart(px.bar(top_ctr, x='Top queries', y='CTR'))
    with col4:
        bottom_ctr = queries_df.nsmallest(10, 'CTR')[['Top queries', 'CTR']]
        st.plotly_chart(px.bar(bottom_ctr, x='Top queries', y='CTR'))

    # Correlation matrix
    st.subheader("📈 Correlation Matrix")
    correlation_matrix = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']].corr()
    st.plotly_chart(px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix'))

    # Anomaly detection with Isolation Forest
    st.subheader("🚨 Anomaly Detection")
    features = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']]
    
    # Optional standardization for better performance
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    contamination = st.slider("Select contamination level (outlier proportion)", min_value=0.005, max_value=0.1, value=0.01, step=0.005)

    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    queries_df['anomaly'] = iso_forest.fit_predict(scaled_features)

    anomalies = queries_df[queries_df['anomaly'] == -1]
    normal = queries_df[queries_df['anomaly'] == 1]

    st.markdown(f"**Total anomalies detected:** {len(anomalies)}")
    st.dataframe(anomalies[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])

    st.subheader("📌 Compare Anomalies vs Normal Queries")
    fig_compare = px.scatter(queries_df, x='Impressions', y='CTR', color=queries_df['anomaly'].map({1:'Normal', -1:'Anomaly'}),
                             hover_data=['Top queries', 'Clicks', 'Position'])
    st.plotly_chart(fig_compare)

else:
    st.warning("Please upload a CSV file with columns: Top queries, Clicks, Impressions, CTR, Position")
