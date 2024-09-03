# Customer Segmentation with K-Means and PCA

This repository contains a Streamlit application for performing customer segmentation using **K-Means Clustering** and **Principal Component Analysis (PCA)**. The app allows users to upload their datasets, adjust clustering parameters, and visualize the results interactively.

## Features

- **K-Means Clustering**: Partitions customer data into distinct segments based on similarity.
- **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining the most important features of the dataset.
- **Data Exploration**: View raw data samples and summary statistics.
- **Correlation Heatmap**: Visualizes correlations between features with color annotations.
- **PCA Explained Variance**: Displays the cumulative explained variance of PCA components.
- **Elbow Method Plot**: Helps determine the optimal number of clusters by visualizing the Within-Cluster Sum of Squares (WCSS).
- **Cluster Scatter Plot**: Visualizes clusters in a 2D space using selected PCA components.

## Getting Started

### Prerequisites

To run this app, you'll need Python 3.7 or higher. Install the required packages by running:

```bash
pip install -r requirements.txt
