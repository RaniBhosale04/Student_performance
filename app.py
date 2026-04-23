from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model during application startup
MODEL_PATH = 'model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    """AWS ELB/Health check endpoint"""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model failed to load"}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500
        
    try:
        data = request.get_json()
        
        # Ensure 'features' key exists in the JSON payload
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON payload'}), 400
            
        # Extract features and reshape for scikit-learn (1 sample, n features)
        # Expected to be a list of 9 elements based on your pickle file
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return the result (converted to standard Python types for JSON serialization)
        return jsonify({'prediction': int(prediction[0])}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run in debug mode if executed directly (not recommended for production)
    app.run(host='0.0.0.0', port=5000, debug=True)
