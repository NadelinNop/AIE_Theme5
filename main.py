import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import dump, load

# Load the cleaned CSV file
file_path = 'clean_data.csv'
df = pd.read_csv(file_path)

# 1. Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# 2. Define a subset of columns that should uniquely identify a record
unique_cols = ['time', 'latitude', 'longitude', 'svr1', 'svr2', 'svr3', 'svr4', 'square_id']

# 2a. Check for duplicates based on this unique identifier set
duplicates = df.duplicated(subset=unique_cols, keep=False)
print("Number of duplicates found:", duplicates.sum())

# Display some of the duplicate entries to manually inspect them
if duplicates.sum() > 0:
    print(df[duplicates].sort_values(by='time').head())

# Remove duplicates if confirmed they are actual redundancies
df_unique = df.drop_duplicates(subset=unique_cols)

# 3. Validate GPS coordinates (assuming invalid values are 99.999)
invalid_gps = df[(df['latitude'] == 99.999) | (df['longitude'] == 99.999)].shape[0]
print("\nNumber of Invalid GPS Coordinates: ", invalid_gps)

# 4. Check data types (ensure numeric columns are numeric)
print("\nData Types:\n", df_unique.dtypes)

# 5. Statistical summary to check for outliers or abnormal values
print("\nStatistical Summary:\n", df_unique.describe())
df['mean_latency'] = df[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)
df['total_bitrate'] = df['upload_bitrate_mbits/sec'] + df['download_bitrate_rx_mbits/sec']

# 6. Custom condition: Check if any column has zero or negative values (for specific columns like bitrate or latency)
columns_to_check = ['svr1', 'svr2', 'svr3', 'svr4', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']
for col in columns_to_check:
    invalid_values = df_unique[df_unique[col] <= 0].shape[0]
    print(f"\nNumber of invalid (<= 0) values in {col}: {invalid_values}")

# Step to handle invalid values:

# Option 1: Replace zero/negative values with NaN for later handling
df_unique['upload_bitrate_mbits/sec'].replace(0, pd.NA, inplace=True)
df_unique['download_bitrate_rx_mbits/sec'].replace(0, pd.NA, inplace=True)

# Option 2: Remove rows with zero/negative values
df_ = df_unique[(df_unique['svr1'] > 0) & (df_unique['svr2'] > 0) &
                (df_unique['svr3'] > 0) & (df_unique['svr4'] > 0) &
                (df_unique['upload_bitrate_mbits/sec'] > 0) &
                (df_unique['download_bitrate_rx_mbits/sec'] > 0)]

# Normalize the features for clustering
scaler = StandardScaler()
features_to_scale = ['mean_latency', 'total_bitrate', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# 7. Apply PCA for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[features_to_scale])

# Add PCA components to the dataframe for plotting
df['pca1'] = df_pca[:, 0]
df['pca2'] = df_pca[:, 1]

# 8. Apply KMeans Clustering
num_clusters = 9  # You can adjust this number
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)

# Train the KMeans model
df['cluster'] = kmeans.fit_predict(df[features_to_scale])

# Save the trained model
model_file = 'kmeans_model.joblib'
dump(scaler, 'scaler.joblib')
dump(pca, 'pca.joblib')

dump(kmeans, model_file)
print(f"KMeans model saved as {model_file}")


# Now, plotting the PCA of clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
plt.title('PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()
# 10. Evaluate the Model
# Inertia (lower is better)
print(f"Inertia: {kmeans.inertia_}")
import numpy as np

# Define the sample size, e.g., 10% of the data
sample_size = int(len(df) * 0.1)
# Ensure the sample is not too small; it should be representative and allow for silhouette calculation
sample_size = max(sample_size, 2*num_clusters)  # At least twice the number of clusters

# Randomly sample the data
sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
df_sample = df.loc[sample_indices, features_to_scale]

# Fit KMeans to the sample and calculate the silhouette score
kmeans_sample = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels_sample = kmeans_sample.fit_predict(df_sample)

# Calculate silhouette score on the sampled data
silhouette_sample = silhouette_score(df_sample, cluster_labels_sample)
print(f"Sampled Silhouette Score: {silhouette_sample}")
