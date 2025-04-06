import sklearn
import numpy as np
from typing import Tuple

def score(text: str, model: sklearn.base.BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Score a text using a trained model and determine if it's spam based on a threshold.
    
    Args:
        text: The input text to be classified
        model: A trained sklearn model with predict_proba method
        threshold: The decision threshold for classification
        
    Returns:
        Tuple containing:
            - prediction: True if the text is classified as spam, False otherwise
            - propensity: The probability score for the positive class
    """
    # Ensure text is in the right format (list or array for sklearn models)
    if not isinstance(text, list):
        text = [text]
    
    # Get probability scores from the model
    propensity = model.predict_proba(text)[0, 1]  # Probability of positive class
    
    # Make prediction based on threshold
    prediction = propensity >= threshold
    
    return bool(prediction), float(propensity) 