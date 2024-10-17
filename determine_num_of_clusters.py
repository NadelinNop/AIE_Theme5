from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

# Load the dataset
file_path = 'processed_for_clustering.csv'
df = pd.read_csv(file_path)

# Features to use for clustering
features = ['mean_latency', 'total_bitrate', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']

# Standardize the features for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Initialize lists to store results
inertias = []
silhouette_scores = []

# Define the range of cluster numbers to test
k_range = range(2, 11)  # Testing clusters from 2 to 10

for k in k_range:
    # Apply KMeans for the current number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)

    # Append the inertia for the current number of clusters
    inertias.append(kmeans.inertia_)

    # Calculate the silhouette score (only if there are more than 1 cluster)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    print(f"For n_clusters = {k}, Inertia = {kmeans.inertia_}, Silhouette Score = {silhouette_avg}")

# Plot the Elbow Method graph (Inertia vs Number of Clusters)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Plot the Silhouette Score graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
