"""
Kaggle LLM Classification Competition
Source code package for predicting human preferences in chatbot conversations.

Competition: https://www.kaggle.com/competitions/llm-classification-finetuning
Author: Nisha Gadhe
Date: February 2026
"""

__version__ = "0.1.0"
__author__ = "Nisha Gadhe"
__email__ = "nisha.sonawane@gmail.com"

# Import main functions for easy access
from .features import (
    extract_length_features,
    extract_quality_features,
    extract_all_features
)
from .utils import (
    load_data,
    create_submission,
    validate_probabilities,
    save_submission,
    plot_feature_importance,
    evaluate_predictions
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Feature extraction
    'extract_length_features',
    'extract_quality_features', 
    'extract_all_features',
    
    # Utilities
    'load_data',
    'create_submission',
    'validate_probabilities',
    'save_submission',
    'plot_feature_importance',
    'evaluate_predictions'
]