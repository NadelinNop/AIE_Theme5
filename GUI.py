from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load model, scaler, and PCA (for visualization only)
model = load('kmeans_model.joblib')
scaler = load('scaler.joblib')  # The scaler used for the 4 original features
pca = load('pca.joblib')  # The PCA (used only for visualization, not for prediction)
cluster_descriptions = {
    0: "High latency, moderate download speeds: Zones with poor network performance and congestion.",
    1: "Low latency, moderate download speeds: Areas with good network performance.",
    2: "Very low latency, high download speeds: Zones with excellent network conditions.",
    3: "Moderate latency, low download speeds: Areas with suboptimal performance.",
    4: "Balanced performance with moderate latency and throughput.",
    5: "High download speeds but slightly higher latency than optimal zones.",
    6: "Low latency, low throughput: This cluster represents lightly used zones.",
    7: "Very high throughput but with latency variations.",
    8: "Zones with unpredictable network performance."
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    data = np.array([[float(form_data['latency']), float(form_data['bitrate']),
                      float(form_data['upload']), float(form_data['download'])]])

    # Scale the data (but don't apply PCA for prediction)
    scaled_data = scaler.transform(data)

    # Predict cluster
    predicted_cluster = model.predict(scaled_data)[0]

    # Fetch cluster description
    description = cluster_descriptions.get(predicted_cluster, "No description available for this cluster.")

    # Render the results page
    return render_template('results.html', cluster=predicted_cluster, description=description)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/help')
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True)
