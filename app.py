import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Title and description
st.title("Global Development Measurement Clustering")
st.markdown("""
This app performs clustering on a global development measurement dataset using K-means clustering.
""")

# Load dataset
#@st.cache
def load_data():
    data = pd.read_excel('World_development_mesurement.xlsx')
    return data

data = load_data()

# Data Cleaning
data['GDP'] = data['GDP'].astype(str).str.replace('\$', '', regex=True).str.replace(',', '')
data['GDP'] = pd.to_numeric(data['GDP'], errors='coerce')

data['Health Exp/Capita'] = data['Health Exp/Capita'].astype(str).str.replace('\$', '', regex=True)
data['Health Exp/Capita'] = pd.to_numeric(data['Health Exp/Capita'], errors='coerce')

data['Tourism Inbound'] = data['Tourism Inbound'].astype(str).str.replace('\$', '', regex=True).str.replace(',', '')
data['Tourism Inbound'] = pd.to_numeric(data['Tourism Inbound'], errors='coerce')

data['Tourism Outbound'] = data['Tourism Outbound'].astype(str).str.replace('\$', '', regex=True).str.replace(',', '')
data['Tourism Outbound'] = pd.to_numeric(data['Tourism Outbound'], errors='coerce')

data['Business Tax Rate'] = data['Business Tax Rate'].astype(str).str.replace('%', '', regex=True)
data['Business Tax Rate'] = pd.to_numeric(data['Business Tax Rate'], errors='coerce')

data.drop(['Country'], axis=1, inplace=True)
data.drop(['Number of Records'], axis=1, inplace=True)

# Fill missing values
data.fillna(data.median(), inplace=True)

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA
pca = PCA()
data_pca = pca.fit_transform(scaled_data)

# Sidebar for user input
st.sidebar.header('User Input Parameters')
num_clusters = st.sidebar.slider('Number of clusters', 2, 10, 3)

# K-means Clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans.fit(data_pca)
clusters = kmeans.labels_

# Add clustering results to the original data
data['Cluster'] = clusters

# Visualization
st.subheader('Cluster Visualization')
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters of Global Development Measurements')
st.pyplot(plt)

# Display the silhouette score
silhouette_avg = silhouette_score(data_pca, clusters)
st.write(f'Silhouette Score: {silhouette_avg}')

# Display the data
st.subheader('Clustered Data')
st.write(data)

# Display cluster characteristics
st.subheader('Cluster Characteristics')
for cluster_num in range(num_clusters):
    st.write(f'**Cluster {cluster_num}**')
    cluster_data = data[data['Cluster'] == cluster_num]
    st.write(cluster_data.describe())

# Download link for clustered data
#@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='clustered_data.csv',
    mime='text/csv',
)

# Footer
st.markdown("""
Developed by ranjith and group
""")
