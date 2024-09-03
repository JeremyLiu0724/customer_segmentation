import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Custom style for the Streamlit app
st.markdown("<style> .right-align { float: right; } </style> <p class='right-align'>Streamlit App by <a href='https://www.linkedin.com/in/ziyu-jeremy-liu/'>Jeremy Liu</a></p>", unsafe_allow_html=True)

# Introduction section
st.markdown("# Customer Segmentation with K-Means and PCA")
st.write("""
    Welcome to the Customer Segmentation application! This app utilizes the power of machine learning to help you 
    understand and segment your customer data. The key methods used here are **K-Means Clustering** and **Principal 
    Component Analysis (PCA)** to identify patterns in your data and visualize the clusters effectively. 

    - **K-Means Clustering** helps partition the data into distinct segments based on similarity.
    - **PCA** reduces the dimensionality of the data while retaining the most important features.

    Upload your dataset on the sidebar, adjust the clustering and PCA parameters, and explore the insights provided 
    by the visualizations.
""")

# Default CSV file
default_csv = "segmentation data.csv"

# Move file uploader to sidebar
st.sidebar.header('Upload your dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)

# Cache data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file, index_col=0)

# Use the uploaded file or the default file
if uploaded_file is not None:
    df_segmentation = load_data(uploaded_file)
else:
    df_segmentation = load_data(default_csv)
    st.sidebar.warning(f"No file uploaded. Using default file: {default_csv}")

# Sidebar: Select PCA components
st.sidebar.header('PCA Components')
n_components = st.sidebar.slider('Number of components', min_value=1, max_value=9, value=3)

# Sidebar: Select number of clusters
st.sidebar.header('K-Means Clustering')
n_clusters = st.sidebar.slider('Number of clusters', min_value=1, max_value=10, value=4)

# Standardize data
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)

# Data exploration
st.markdown('### Data Exploration')
show_raw_data = st.checkbox('Show Raw Data Sample')
if show_raw_data:
    st.write(df_segmentation.head())

show_summary_stats = st.checkbox('Show Summary Statistics')
if show_summary_stats:
    st.write(df_segmentation.describe())

# Correlation heatmap using Plotly
st.markdown('### Correlation Heatmap')
corr_matrix = df_segmentation.corr()
fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
# Add annotations for each cell in the heatmap
annotations = []
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        annotations.append(
            go.layout.Annotation(
                x=j, y=i, text=str(round(corr_matrix.iloc[i, j], 2)),
                showarrow=False, font=dict(color="black")
            )
        )
fig.update_layout(annotations=annotations)
st.plotly_chart(fig)

# Explained Variances by Components - using Plotly
st.markdown('### PCA: Explained Variance by Components')
pca = PCA()
pca.fit(segmentation_std)
explained_variance = pca.explained_variance_ratio_

fig = go.Figure(data=go.Scatter(x=list(range(1, len(explained_variance)+1)), 
                                y=explained_variance.cumsum(), 
                                mode='lines+markers', 
                                line=dict(color='royalblue', width=2),
                                marker=dict(size=10)))
fig.update_layout(title='Cumulative Explained Variance by PCA Components',
                  xaxis_title='Number of Components',
                  yaxis_title='Cumulative Explained Variance',
                  template="plotly_dark")
st.plotly_chart(fig)

# Perform PCA and K-Means clustering
pca = PCA(n_components=n_components)
scores_pca = pca.fit_transform(segmentation_std)
kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

# Create a new dataframe with original features and PCA scores as well as assigned clusters
df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_segm_pca_kmeans.columns.values[-n_components:] = [f'Component {i+1}' for i in range(n_components)]
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

# Elbow method for K-Means - using Plotly
st.markdown('### K-Means with PCA Clustering')
wcss = []
for i in range(1, 11):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

fig = go.Figure(data=go.Scatter(x=list(range(1, 11)), 
                                y=wcss, 
                                mode='lines+markers', 
                                line=dict(color='firebrick', width=2),
                                marker=dict(size=10)))
fig.update_layout(title='Elbow Method for Optimal Clusters',
                  xaxis_title='Number of Clusters',
                  yaxis_title='WCSS (Within-Cluster Sum of Squares)',
                  template="plotly_dark")
st.plotly_chart(fig)

# Clusters by PCA components - using Plotly
st.markdown('### Clusters by PCA Components')
x_axis_component = st.selectbox('X-axis', [f'Component {i+1}' for i in range(n_components)], index = 0)
y_axis_component = st.selectbox('Y-axis', [f'Component {i+1}' for i in range(n_components)], index = 1)

fig = px.scatter(df_segm_pca_kmeans, 
                 x=x_axis_component, 
                 y=y_axis_component, 
                 color='Segment K-means PCA',
                 title='Clusters by PCA Components',
                 template="plotly_dark",
                 color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig)

# Footer section with hyperlinks
st.markdown("---")  # Horizontal line separator
st.markdown("### Additional Resources")
st.markdown("""
For more details on segmentation analysis, check out my [Kaggle notebook](https://www.kaggle.com/code/jeremyliu1989/customer-analytics-segmentation).

You can find the source code for this Streamlit app on my [GitHub repository](https://github.com/JeremyLiu0724/customer_segmentation).

If you have any feedback, suggestions, or feature requests, feel free to open an issue in the [GitHub repository](https://github.com/JeremyLiu0724/customer_segmentation/issues). Your input is valuable and will help improve the tool!
""")

# Custom style to position the footer in the lower right corner
st.markdown("""
    <style>
    .footer { 
        position: fixed;
        bottom: 10px;
        right: 10px;
        text-align: right;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>Enjoy Data Science ❤️</p>
    </div>
""", unsafe_allow_html=True)
