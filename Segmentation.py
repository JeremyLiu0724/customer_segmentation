import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Set the title and description of the app
st.title("Customer Segmentation App")
st.markdown("""
This app allows you to perform customer segmentation using hierarchical and flat clustering techniques like PCA and K-means.
Upload your dataset, configure the clustering, and visualize the results.
""")

# Data upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

# If a file is uploaded, read the data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.write("Please upload a CSV file to proceed.")

# Display the first few rows of the dataset
if uploaded_file is not None:
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape of the dataset:", df.shape)

if uploaded_file is not None:
    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

if uploaded_file is not None:
    st.subheader("Clustering Configuration")
    
    # Choose the number of clusters
    num_clusters = st.slider("Select the number of clusters", 2, 10, 4)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Apply PCA if selected
    pca_option = st.checkbox("Apply PCA for dimensionality reduction")
    if pca_option:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        st.write("Explained Variance by Components:", pca.explained_variance_ratio_)
    else:
        df_pca = df_scaled

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(df_pca)
    df['Cluster'] = clusters
    
    st.subheader("Clustering Results")
    st.write("Cluster Assignments:", np.unique(clusters, return_counts=True))
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Clusters Visualization")
    st.pyplot(plt)

