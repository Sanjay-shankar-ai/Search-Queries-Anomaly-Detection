# Search Query Anomaly Detector
A Streamlit-based web app to analyze search query performance data and detect anomalies using an Isolation Forest model. Designed for SEO analysts, it processes the Queries.csv dataset (614 programming-related queries) and provides interactive visualizations and anomaly detection.

## Features
```
Data Input: Upload Queries.csv with columns: Top queries, Clicks, Impressions, CTR, Position.
Exploratory Analysis: Visualizes top words, clicks, impressions, CTR extremes, and correlation matrix.
Anomaly Detection: Identifies outliers using Isolation Forest on numerical features.
Interactivity: Adjust contamination level (0.005–0.1) to tune anomaly sensitivity.
Visualizations: Scatter plots and tables to compare anomalies vs. normal queries.
```

## Dataset

Source: Queries.csv, likely from Google Search Console, with 614 rows of programming and machine learning queries.

Columns: Top queries (e.g., "number guessing game python"), Clicks (48–5,223), Impressions (62–73,380), CTR (0.29%–85.48%), Position (1.0–28.52).

Example:
Top queries,Clicks,Impressions,CTR,Position
number guessing game python,5223,14578,35.83%,1.61
thecleverprogrammer,2809,3456,81.28%,1.02



## Setup Instructions

Clone the Repository:
git clone <repository-url>
cd <repository-folder>


## Install Dependencies:
pip install streamlit pandas plotly scikit-learn


## Run the App:
streamlit run streamlit_app.py


## Access the App: 

Upload Dataset: Use Queries.csv with required columns.


## Requirements

Python: 3.8+
Libraries: streamlit>=1.20.0, pandas>=1.5.0, plotly>=5.10.0, scikit-learn>=1.2.0
OS: Windows, macOS, or Linux

## Usage

Upload Queries.csv via the app interface.
Explore visualizations (e.g., word frequency, correlation matrix).
Adjust the contamination slider to detect anomalies (e.g., high-CTR queries like "thecleverprogrammer.com").
View anomaly table and scatter plot for insights.

