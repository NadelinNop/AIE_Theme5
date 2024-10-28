from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for rendering to files
import seaborn as sns
from io import BytesIO
import base64
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load model, scaler, and PCA (for visualization only)
model = load('kmeans_model.joblib')
scaler = load('scaler.joblib')  # The scaler used for the 4 original features
pca = load('pca.joblib')  # The PCA (used only for visualization, not for prediction)

# Load dataset for visualization purposes
df = pd.read_csv('processed_with_clusters.csv')
features_to_scale = ['mean_latency', 'total_bitrate', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec','application_data']

long_cluster_descriptions = {
    0: "This zone experiences the most stable but moderate network performance. While the upload speeds tend to be a bit lower, the overall download speeds are consistent, making it suitable for light browsing and general use.",
    1: "Customers in this zone will experience faster upload and download speeds compared to other zones. It’s ideal for people who frequently upload content, engage in video conferencing, or stream videos. The latency here is balanced, providing a smooth internet experience.",
    2: "Network performance in this zone is stable with moderate speeds for both uploads and downloads. It’s suitable for everyday activities like social media, video streaming, and online shopping without noticeable delays.",
    3: "This zone has some fluctuations in download speeds, but for most users, the internet experience will be smooth for regular online activities like web browsing, video calls, and downloading content.",
    4: "Similar to Zone 1, this zone provides reliable upload speeds, making it a good choice for users who regularly share large files or stream content. However, it may show slightly lower performance in overall network throughput compared to Zone 1.",
    5: "This zone has the highest upload capacity, making it perfect for heavy upload users, such as content creators or those who frequently upload large files to the cloud. The download speeds are steady, ensuring a smooth experience for other activities.",
    6: "This zone has a bit more latency, meaning there could be slight delays in some online activities, particularly gaming or real-time video communication. However, the total network capacity is high, making it great for users with multiple devices connected simultaneously.",
    7: "In this zone, users may experience occasional fluctuations in download speeds, but the overall internet experience remains smooth for most activities. It’s a reliable zone for daily use, though large downloads might take longer during peak times.",
    8: "This zone has a unique network profile with slightly lower overall speeds, both for uploads and downloads. It’s ideal for lighter internet users who don’t require high-speed performance but need reliable connectivity for basic tasks like email, web browsing, and social media."
}

cluster_labels = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
    5: "Cluster 5",
    6: "Cluster 6",
    7: "Cluster 7",
    8: "Cluster 8"
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    # Save the plot as base64 JPG for rendering in the HTML template
    def save_plot_to_base64():
        img = BytesIO()
        plt.savefig(img, format='jpg')  # Save as JPG
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    form_data = request.form
    total_bitrate = float(form_data['upload']) + float(form_data['download'])
    rounded_total = round(total_bitrate, 2)

    # Collect user input and scale the data
    input_data = {
        'mean_latency': [float(form_data['latency'])],
        'total_bitrate': [total_bitrate],
        'upload_bitrate_mbits/sec': [float(form_data['upload'])],
        'download_bitrate_rx_mbits/sec': [float(form_data['download'])],
        'application_data' : [float(form_data['appdata'])]
    }

    data = pd.DataFrame(input_data)
    scaled_data = scaler.transform(data)  # Scale the user input
    predicted_cluster = model.predict(scaled_data)[0]
    
    # Apply PCA to the entire dataset (use already scaled data)
    pca_features = pca.transform(df[features_to_scale])  # Use pre-fitted PCA
    df['PCA1'], df['PCA2'] = pca_features[:, 0], pca_features[:, 1]

    # Map the cluster numbers to the descriptive labels in the dataframe
    df['cluster_label'] = df['cluster'].map(cluster_labels)
    
    # Fetch cluster description
    description = long_cluster_descriptions.get(predicted_cluster, "No description available for this cluster.")
    edge_color = "black"
    user_input_size = 200
    colors = sns.color_palette('Set1', len(df['cluster'].unique()))
    
    # Latency Comparison Plot 
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster', y='mean_latency', palette=colors)
    plt.scatter(predicted_cluster, scaled_data[0][0], s=user_input_size, color='red', 
                edgecolor=edge_color, zorder=5, label='User Input')
    # plt.title('Latency Comparison (ms)')
    latency_img_base64 = save_plot_to_base64()  # Save latency plot
    plt.close()

    # Upload Bitrate Comparison Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster', y='upload_bitrate_mbits/sec',  palette=colors)
    plt.scatter(predicted_cluster, scaled_data[0][2], s=user_input_size, color='red', 
                edgecolor=edge_color, zorder=5, label='User Input')
    # plt.title('Upload Bitrate Comparison (Mbps)')
    upload_img_base64 = save_plot_to_base64()  # Save upload bitrate plot
    plt.close()

    # Download Bitrate Comparison Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster', y='download_bitrate_rx_mbits/sec', palette=colors)
    plt.scatter(predicted_cluster, scaled_data[0][3], s=user_input_size, color='red', 
                edgecolor=edge_color, zorder=5, label='User Input')
    # plt.title('Download Bitrate Comparison (Mbps)')
    download_img_base64 = save_plot_to_base64()  # Save download bitrate plot
    plt.close()

    # Plot PCA scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster_label', palette=colors)

    # Now apply the same PCA transform to the user input
    user_pca = pca.transform(scaled_data)  # Apply PCA to the scaled user input

    # Plot the user's input on the PCA plot
    plt.scatter(user_pca[0, 0], user_pca[0, 1], s=user_input_size, color='red', edgecolor=edge_color, zorder=5, label='User Input')
    # plt.title('PCA Scatter Plot (Clusters)')
    plt.legend(loc="center right") 
    pca_img_base64 = save_plot_to_base64()
    plt.close()  # Close the plot after saving to avoid memory leaks
    
        # Plot PCA scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='cluster_label', palette=colors)
    # plt.title('PCA Scatter Plot (Clusters)')
    plt.legend(loc="center right") 
    latlong_img_base64 = save_plot_to_base64()
    plt.close()  # Close the plot after saving to avoid memory leaks

    # Render the results page with the PCA scatter plot
    return render_template('results.html', 
                        cluster=predicted_cluster, 
                        description=description, 
                        latency=form_data['latency'], 
                        bitrate=rounded_total, 
                        upload=form_data['upload'], 
                        download=form_data['download'],
                        latency_image=latency_img_base64,
                        upload_image=upload_img_base64,
                        download_image=download_img_base64,
                        pca_image=pca_img_base64, latlong_image=latlong_img_base64)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)