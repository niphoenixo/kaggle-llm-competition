"""
Utility functions for Kaggle LLM Classification Competition.

This module provides helper functions for:
- Data loading and preprocessing
- Submission file creation and validation
- Model evaluation and visualization
- Probability calibration and checks
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(data_path: str = '/kaggle/input/llm-classification-finetuning/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from Kaggle competition.
    
    Args:
        data_path: Path to competition data directory
        
    Returns:
        Tuple of (train_df, test_df)
    
    Example:
        >>> train, test = load_data()
        >>> print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    """
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"""
        Training data not found at {train_path}
        
        Please ensure:
        1. You've joined the competition: https://www.kaggle.com/competitions/llm-classification-finetuning
        2. Created notebook FROM competition page
        3. Added competition data via '+ Add Data' button
        """)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Training columns: {list(train_df.columns)}")
    print(f"   Test columns: {list(test_df.columns)}")
    
    return train_df, test_df


def get_target_stats(train_df: pd.DataFrame) -> Dict:
    """
    Calculate target variable statistics from training data.
    
    Args:
        train_df: Training dataframe with winner_model_a, winner_model_b, winner_tie columns
        
    Returns:
        Dictionary with target statistics
    """
    stats = {
        'mean_model_a': train_df['winner_model_a'].mean(),
        'mean_model_b': train_df['winner_model_b'].mean(),
        'mean_tie': train_df['winner_tie'].mean(),
        'std_model_a': train_df['winner_model_a'].std(),
        'std_model_b': train_df['winner_model_b'].std(),
        'std_tie': train_df['winner_tie'].std(),
        'median_model_a': train_df['winner_model_a'].median(),
        'median_model_b': train_df['winner_model_b'].median(),
        'median_tie': train_df['winner_tie'].median(),
    }
    
    print("\n📊 Target Variable Statistics:")
    print(f"   Model A wins: mean={stats['mean_model_a']:.4f}, std={stats['std_model_a']:.4f}")
    print(f"   Model B wins: mean={stats['mean_model_b']:.4f}, std={stats['std_model_b']:.4f}")
    print(f"   Tie: mean={stats['mean_tie']:.4f}, std={stats['std_tie']:.4f}")
    
    return stats


# ============================================================================
# SUBMISSION CREATION & VALIDATION
# ============================================================================

def create_submission(
    test_df: pd.DataFrame,
    predictions_a: Union[List, np.ndarray],
    predictions_b: Union[List, np.ndarray],
    predictions_tie: Union[List, np.ndarray],
    method_name: str = "baseline"
) -> pd.DataFrame:
    """
    Create submission dataframe with probability predictions.
    
    Args:
        test_df: Test dataframe with 'id' column
        predictions_a: Predicted probabilities for model_a winning
        predictions_b: Predicted probabilities for model_b winning
        predictions_tie: Predicted probabilities for tie
        method_name: Name of the prediction method (for logging)
        
    Returns:
        Submission dataframe ready for Kaggle
    
    Example:
        >>> probs_a = [0.6, 0.2, 0.33]
        >>> probs_b = [0.2, 0.6, 0.33]
        >>> probs_tie = [0.2, 0.2, 0.34]
        >>> submission = create_submission(test_df, probs_a, probs_b, probs_tie)
    """
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': predictions_a,
        'winner_model_b': predictions_b,
        'winner_tie': predictions_tie
    })
    
    # Validate probabilities
    is_valid, message = validate_probabilities(submission)
    if not is_valid:
        print(f"⚠️  Warning: {message}")
        print("   Attempting to fix probabilities...")
        submission = calibrate_probabilities(submission)
    else:
        print(f"✅ {method_name}: Valid submission created!")
    
    return submission


def validate_probabilities(submission_df: pd.DataFrame, rtol: float = 1e-6) -> Tuple[bool, str]:
    """
    Validate that probabilities sum to 1 for each row.
    
    Args:
        submission_df: Submission dataframe with probability columns
        rtol: Relative tolerance for floating point comparison
        
    Returns:
        Tuple of (is_valid, message)
    """
    sums = (
        submission_df['winner_model_a'] + 
        submission_df['winner_model_b'] + 
        submission_df['winner_tie']
    )
    
    all_close = np.allclose(sums, 1.0, rtol=rtol)
    min_sum = sums.min()
    max_sum = sums.max()
    
    if all_close:
        return True, f"All probabilities sum to 1 (min={min_sum:.6f}, max={max_sum:.6f})"
    else:
        return False, f"Probabilities don't sum to 1 (min={min_sum:.6f}, max={max_sum:.6f})"


def calibrate_probabilities(submission_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrate probabilities to ensure they sum to 1.
    
    Args:
        submission_df: Submission dataframe with uncalibrated probabilities
        
    Returns:
        Calibrated submission dataframe
    """
    df = submission_df.copy()
    
    # Calculate sums
    sums = df['winner_model_a'] + df['winner_model_b'] + df['winner_tie']
    
    # Calibrate each row
    df['winner_model_a'] = df['winner_model_a'] / sums
    df['winner_model_b'] = df['winner_model_b'] / sums
    df['winner_tie'] = df['winner_tie'] / sums
    
    # Verify calibration
    new_sums = df['winner_model_a'] + df['winner_model_b'] + df['winner_tie']
    print(f"✅ Probabilities calibrated: max deviation = {abs(new_sums - 1.0).max():.10f}")
    
    return df


def save_submission(
    submission_df: pd.DataFrame,
    filename: str = 'submission.csv',
    sub_dir: str = 'submissions'
) -> str:
    """
    Save submission file and create a dated copy.
    
    Args:
        submission_df: Submission dataframe
        filename: Name of submission file
        sub_dir: Directory to save submissions
        
    Returns:
        Path to saved submission file
    """
    # Create submissions directory if it doesn't exist
    os.makedirs(sub_dir, exist_ok=True)
    
    # Save main submission file
    main_path = os.path.join('/kaggle/working', filename)
    submission_df.to_csv(main_path, index=False)
    
    # Save dated copy in submissions folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(sub_dir, f'submission_{timestamp}.csv')
    submission_df.to_csv(archive_path, index=False)
    
    # Also save as first_submission.csv if it's the first one
    first_submission_path = os.path.join(sub_dir, 'first_submission.csv')
    if not os.path.exists(first_submission_path):
        submission_df.to_csv(first_submission_path, index=False)
        print(f"💾 Saved as first submission: {first_submission_path}")
    
    print(f"💾 Submission saved to: {main_path}")
    print(f"📦 Archived to: {archive_path}")
    print(f"   File size: {os.path.getsize(main_path):,} bytes")
    
    return main_path


# ============================================================================
# MODEL EVALUATION & METRICS
# ============================================================================

def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        y_true: Ground truth probabilities
        y_pred: Predicted probabilities
        verbose: Print results if True
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
    
    # Prepare data
    y_true_array = y_true[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    y_pred_array = y_pred[['winner_model_a', 'winner_model_b', 'winner_tie']].values
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(y_true_array, y_pred_array),
        'mse': mean_squared_error(y_true_array, y_pred_array),
        'mae': mean_absolute_error(y_true_array, y_pred_array)
    }
    
    if verbose:
        print("\n📈 Evaluation Metrics:")
        print(f"   Log Loss: {metrics['log_loss']:.6f}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
    
    return metrics


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for cross-validation scores.
    
    Args:
        scores: List of evaluation scores
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    import scipy.stats as st
    
    n = len(scores)
    mean = np.mean(scores)
    sem = st.sem(scores)
    margin = sem * st.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - margin, mean + margin


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(
    feature_names: List[str],
    importance_values: List[float],
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance bar chart.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), [importance_values[i] for i in indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not installed. Skipping visualization.")


def plot_prediction_distribution(
    submission_df: pd.DataFrame,
    title: str = "Prediction Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of predicted probabilities.
    
    Args:
        submission_df: Submission dataframe with predictions
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        columns = ['winner_model_a', 'winner_model_b', 'winner_tie']
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        for ax, col, color in zip(axes, columns, colors):
            sns.histplot(submission_df[col], kde=True, ax=ax, color=color, bins=30)
            ax.set_title(f'{col.replace("_", " ").title()}')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Frequency')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not installed. Skipping visualization.")


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def log_submission(
    score: float,
    approach: str,
    features: List[str],
    model_type: str,
    notes: str = "",
    log_file: str = "reports/progress_log.json"
):
    """
    Log competition progress to JSON file.
    
    Args:
        score: Kaggle log loss score
        approach: Description of approach
        features: List of features used
        model_type: Type of model used
        notes: Additional notes
        log_file: Path to log file
    """
    # Create log entry
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": score,
        "approach": approach,
        "features": features,
        "model_type": model_type,
        "notes": notes,
        "submission_id": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    # Append new entry
    logs.append(entry)
    
    # Save logs
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"📝 Progress logged: Score={score:.4f}, Approach={approach}")


# ============================================================================
# ENSEMBLE UTILITIES
# ============================================================================

def ensemble_predictions(
    predictions_list: List[pd.DataFrame],
    weights: Optional[List[float]] = None,
    method: str = 'weighted'
) -> pd.DataFrame:
    """
    Ensemble multiple submission files.
    
    Args:
        predictions_list: List of submission dataframes
        weights: List of weights for each prediction (default: equal)
        method: 'weighted' or 'average'
        
    Returns:
        Ensembled submission dataframe
    """
    if not predictions_list:
        raise ValueError("Empty predictions list")
    
    n = len(predictions_list)
    
    if weights is None:
        weights = [1.0/n] * n
    else:
        weights = np.array(weights) / np.sum(weights)
    
    # Initialize with first prediction
    ensemble = predictions_list[0].copy()
    
    # Weighted average of probabilities
    prob_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    
    for col in prob_cols:
        ensemble[col] = np.zeros_like(ensemble[col], dtype=float)
        for pred, weight in zip(predictions_list, weights):
            ensemble[col] += weight * pred[col].values
    
    # Ensure probabilities sum to 1
    ensemble = calibrate_probabilities(ensemble)
    
    print(f"🎯 Ensemble created with {n} models, method={method}")
    print(f"   Weights: {[round(w, 3) for w in weights]}")
    
    return ensemble


# ============================================================================
# FEATURE ENGINEERING WRAPPER
# ============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Dataframe with response_a and response_b columns
        
    Returns:
        Dataframe with engineered features
    """
    from .features import extract_all_features
    
    features_list = []
    
    for idx, row in df.iterrows():
        features = extract_all_features(row['response_a'], row['response_b'])
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Add original IDs
    if 'id' in df.columns:
        features_df['id'] = df['id'].values
    
    print(f"✅ Feature engineering complete: {features_df.shape[1]} features created")
    
    return features_df


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def setup_kaggle_environment():
    """
    Setup Kaggle environment and check requirements.
    """
    print("🔧 Checking Kaggle environment...")
    
    # Check if running in Kaggle
    in_kaggle = os.path.exists('/kaggle')
    print(f"   Running in Kaggle: {in_kaggle}")
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_name} ✓")
        else:
            print("   GPU: Not available (using CPU)")
    except:
        print("   PyTorch not installed")
    
    # Check competition data
    data_exists = os.path.exists('/kaggle/input/llm-classification-finetuning')
    if data_exists:
        print("   Competition data: ✓")
    else:
        print("   Competition data: ✗ (please add via '+ Add Data')")
    
    print("\n✅ Environment check complete")


# ============================================================================
# MAIN EXECUTION GUARD
# ============================================================================

if __name__ == "__main__":
    print("🔬 Kaggle LLM Competition - Utility Functions")
    print(f"   Version: {__version__}")
    print(f"   Author: {__author__}")
    print("\n📚 Available functions:")
    
    functions = [
        'load_data', 'get_target_stats',
        'create_submission', 'validate_probabilities', 'calibrate_probabilities', 'save_submission',
        'evaluate_predictions', 'calculate_confidence_interval',
        'plot_feature_importance', 'plot_prediction_distribution',
        'log_submission', 'ensemble_predictions', 'prepare_features',
        'setup_kaggle_environment'
    ]
    
    for func in functions:
        print(f"   • {func}")