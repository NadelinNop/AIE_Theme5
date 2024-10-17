import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load the dataset
file_path = 'final_cleaned_data.csv'
df = pd.read_csv(file_path)

# Sampling 5% of the dataset for a more drastic reduction
sampled_df = df.sample(frac=0.05, random_state=42)

# Calculate mean latency and total bitrate if columns are available
if all(x in sampled_df.columns for x in ['svr1', 'svr2', 'svr3', 'svr4']):
    sampled_df['mean_latency'] = sampled_df[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)
else:
    raise ValueError("Required columns for 'mean_latency' calculation are missing!")

if all(x in sampled_df.columns for x in ['upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']):
    sampled_df['total_bitrate'] = sampled_df['upload_bitrate_mbits/sec'] + sampled_df['download_bitrate_rx_mbits/sec']
else:
    raise ValueError("Required columns for 'total_bitrate' calculation are missing!")

# Scale features
features_to_scale = ['mean_latency', 'total_bitrate']
sampled_df[features_to_scale] = StandardScaler().fit_transform(sampled_df[features_to_scale])

# Clustering with different linkage method
num_clusters = 9
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=5)
agglo = AgglomerativeClustering(n_clusters=num_clusters, linkage='average')  # Changed linkage method

# Fit models on sampled data
sampled_df['kmeans_cluster'] = kmeans.fit_predict(sampled_df[features_to_scale])
sampled_df['dbscan_cluster'] = dbscan.fit_predict(sampled_df[features_to_scale])
sampled_df['agglo_cluster'] = agglo.fit_predict(sampled_df[features_to_scale])

# Metrics
def calculate_clustering_metrics(data, labels):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        return silhouette, davies_bouldin, calinski_harabasz
    return None, None, None

kmeans_metrics = calculate_clustering_metrics(sampled_df[features_to_scale], sampled_df['kmeans_cluster'])
dbscan_metrics = calculate_clustering_metrics(sampled_df[features_to_scale], sampled_df['dbscan_cluster'])
agglo_metrics = calculate_clustering_metrics(sampled_df[features_to_scale], sampled_df['agglo_cluster'])

# Print results
print("KMeans Metrics:", kmeans_metrics)
print("DBSCAN Metrics:", dbscan_metrics)
print("Agglomerative Metrics:", agglo_metrics)
