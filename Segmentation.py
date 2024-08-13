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
st.title("Customer Segmentation Solver")
st.markdown("""
This app can help you get segmentation solution using flat clustering techniques like PCA and K-means.
Upload your dataset, configure the clustering, and visualize the results.
""")

# Data upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

# If a file is uploaded, read the data
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_segmentation = pd.read_csv(uploaded_file, index_col=0)
    else:
        df_segmentation = pd.read_excel(uploaded_file, index_col=0)

    # Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.write(df_segmentation.head())
    st.write("Shape of the dataset:", df_segmentation.shape)

    st.subheader("Summary Statistics")
    st.write(df_segmentation.describe())

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_segmentation.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)
    st.pyplot(plt)

    # Scatter Plot Section
    st.subheader("Scatter Plot of Two Features")

    # Allow the user to select features for the scatter plot
    feature_x = st.selectbox("Select feature for X-axis", df_segmentation.columns)
    feature_y = st.selectbox("Select feature for Y-axis", df_segmentation.columns)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_segmentation[feature_x].values, df_segmentation[feature_y].values, alpha=0.5)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Scatter Plot: {feature_x} vs {feature_y}")
    st.pyplot(plt)

    st.subheader("Clustering Configuration")

    # Standardize the data
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(df_segmentation)

    # Compute WCSS (Within-Cluster Sum of Square) for different numbers of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(segmentation_std)
        wcss.append(kmeans.inertia_)

    # Plot WCSS to help determine the optimal number of clusters
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for K-means Clustering')
    st.pyplot(plt)

    # Choose the number of clusters with a slider
    num_clusters = st.slider("Select the number of clusters", 2, 10, 4)

    # Option to apply PCA for dimensionality reduction
    pca_option = st.checkbox("Apply PCA for dimensionality reduction")
    if pca_option:
        pca = PCA(n_components=3)
        df_pca = pca.fit_transform(segmentation_std)
        st.write("Explained Variance by Components:", pca.explained_variance_ratio_)
    else:
        df_pca = segmentation_std

    # Perform K-means clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(df_pca)
    df_segmentation['Cluster'] = clusters  # Add the cluster labels to the original DataFrame

    # Display clustering results
    st.subheader("Clustering Results")
    st.write("Cluster Assignments:", np.unique(clusters, return_counts=True))

    # Plot the clusters (assuming 2D visualization from PCA or standardized data)
    plt.figure(figsize=(10, 8))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.xlabel("Component 1" if pca_option else "Feature 1")
    plt.ylabel("Component 2" if pca_option else "Feature 2")
    plt.title("Clusters Visualization")
    st.pyplot(plt)

else:
    st.write("Please upload a CSV or Excel file to proceed.")
