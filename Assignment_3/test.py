import pytest
import os
import time
import signal
import subprocess
import requests
import joblib
import numpy as np
from score import score

# Load the model for testing
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

class TestScore:
    """Tests for the score function"""
    
    def test_smoke(self):
        """Smoke test - check if the function runs without crashing"""
        result = score("This is a test", model, 0.5)
        assert result is not None
    
    def test_format(self):
        """Format test - check input/output formats and types"""
        prediction, propensity = score("This is a test", model, 0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)
    
    def test_prediction_values(self):
        """Test that prediction is binary (0 or 1 when converted to int)"""
        prediction, _ = score("This is a test", model, 0.5)
        assert prediction in [True, False]
    
    def test_propensity_range(self):
        """Test that propensity is between 0 and 1"""
        _, propensity = score("This is a test", model, 0.5)
        assert 0 <= propensity <= 1
    
    def test_threshold_zero(self):
        """Test that threshold 0 always gives prediction 1"""
        prediction, _ = score("This is a test", model, 0)
        assert prediction is True
    
    def test_threshold_one(self):
        """Test that threshold 1 always gives prediction 0"""
        prediction, _ = score("This is a test", model, 1)
        assert prediction is False
    
    def test_obvious_spam(self):
        """Test that obvious spam text gives prediction 1"""
        obvious_spam = "URGENT: Buy now! Free Viagra, Casino, Make money fast!!!"
        prediction, _ = score(obvious_spam, model, 0.5)
        assert prediction is True
    
    def test_obvious_ham(self):
        """Test that obvious non-spam text gives prediction 0"""
        obvious_ham = "Hi John, can we schedule the meeting for tomorrow at 2pm? Thanks"
        prediction, _ = score(obvious_ham, model, 0.5)
        assert prediction is False

class TestFlask:
    """Integration tests for the Flask application"""
    
    @pytest.fixture
    def flask_app(self):
        """Fixture to start and stop the Flask app for testing"""
        # Start Flask app as a separate process
        process = subprocess.Popen(["python", "app.py"])
        
        # Wait for the server to start
        time.sleep(2)
        
        yield process
        
        # Clean up: stop the Flask app
        process.send_signal(signal.SIGTERM)
        process.wait()
    
    def test_flask_endpoint(self, flask_app):
        """Test the /score endpoint"""
        # Test with a non-spam message
        response = requests.post(
            "http://localhost:5000/score",
            json={"text": "Meeting scheduled for tomorrow"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], int)
        assert isinstance(data["propensity"], float)
        assert 0 <= data["propensity"] <= 1
        
        # Test with a spam message
        response = requests.post(
            "http://localhost:5000/score",
            json={"text": "FREE MONEY! CLICK HERE! VIAGRA DISCOUNT!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1  # Should be classified as spam
        
        # Test with custom threshold
        response = requests.post(
            "http://localhost:5000/score",
            json={"text": "Normal text", "threshold": 0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1  # With threshold 0, everything should be classified as spam