from flask import Flask, request, jsonify
import joblib
import os
from score import score

app = Flask(__name__)

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

# Default threshold
THRESHOLD = 0.5

@app.route('/score', methods=['POST'])
def score_endpoint():
    """Endpoint to score text for spam detection"""
    # Get JSON data from the request
    data = request.get_json()
    
    # Validate input
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    
    # Get optional threshold from request
    threshold = data.get('threshold', THRESHOLD)
    
    # Score the text
    prediction, propensity = score(text, model, threshold)
    
    # Return the result
    return jsonify({
        "prediction": int(prediction),  # Convert to int for JSON compatibility
        "propensity": propensity
    })

if __name__ == '__main__':
    app.run(debug=True) 