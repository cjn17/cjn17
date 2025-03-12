#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates predictions to identify the properties most likely to be listed for sale.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the configuration file"""
    try:
        with open(project_root / 'config' / 'config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_model():
    """
    Load the latest trained model
    
    Returns:
        object: Trained model
    """
    models_dir = project_root / 'models'
    
    # Find the latest model file
    files = list(models_dir.glob("*_model_*.joblib"))
    if not files:
        logger.error("No model files found")
        return None
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading model from {latest_file}")
    
    try:
        model = joblib.load(latest_file)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_prediction_data():
    """
    Load the processed data for making predictions
    
    Returns:
        pandas.DataFrame: DataFrame containing processed data
    """
    processed_dir = project_root / 'data' / 'processed'
    
    # Find the latest model data file
    files = list(processed_dir.glob("model_data_*.csv"))
    if not files:
        logger.error("No model data files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading prediction data from {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df
    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        return pd.DataFrame()

def prepare_features(df):
    """
    Prepare the feature data for prediction
    
    Args:
        df (pandas.DataFrame): DataFrame containing processed data
        
    Returns:
        tuple: X_pred (feature DataFrame), property_info (property details DataFrame)
    """
    logger.info(f"Preparing features for {len(df)} properties")
    
    # Select property identifier columns
    property_info = df[['property_id', 'address', 'postcode_area']].copy()
    
    # Select feature columns (excluding property identifiers)
    feature_cols = [col for col in df.columns if col not in ['property_id', 'address', 'postcode_area']]
    
    if not feature_cols:
        logger.error("No feature columns found in the data")
        return pd.DataFrame(), property_info
    
    X_pred = df[feature_cols].copy()
    
    # Handle missing values
    X_pred = X_pred.fillna(X_pred.median())
    
    return X_pred, property_info

def generate_predictions(model, X_pred, property_info, config):
    """
    Generate predictions for the properties
    
    Args:
        model: Trained model
        X_pred (pandas.DataFrame): Feature DataFrame
        property_info (pandas.DataFrame): Property details DataFrame
        config (dict): Configuration dictionary
        
    Returns:
        pandas.DataFrame: DataFrame with predictions
    """
    if model is None or X_pred.empty:
        logger.error("Model or prediction data is not available")
        return pd.DataFrame()
    
    logger.info("Generating predictions")
    
    # Generate prediction probabilities
    try:
        y_pred_proba = model.predict_proba(X_pred)[:, 1]
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return pd.DataFrame()
    
    # Add predictions to property info
    result_df = property_info.copy()
    result_df['prediction_probability'] = y_pred_proba
    
    # Sort by probability in descending order
    result_df = result_df.sort_values('prediction_probability', ascending=False)
    
    return result_df

def filter_top_properties_by_postcode(predictions_df, config):
    """
    Filter the top N properties most likely to be listed for each postcode
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        config (dict): Configuration dictionary
        
    Returns:
        pandas.DataFrame: DataFrame with top N properties per postcode
    """
    top_n = config['model']['top_n_predictions']
    postcodes = config['postcodes']
    
    logger.info(f"Filtering top {top_n} properties per postcode")
    
    # Initialize empty DataFrame for results
    top_properties = pd.DataFrame()
    
    # Get top N properties for each postcode
    for postcode in postcodes:
        postcode_df = predictions_df[predictions_df['postcode_area'] == postcode]
        
        if len(postcode_df) > 0:
            top_postcode = postcode_df.head(top_n)
            top_properties = pd.concat([top_properties, top_postcode])
            logger.info(f"Selected {len(top_postcode)} properties for postcode {postcode}")
        else:
            logger.warning(f"No properties found for postcode {postcode}")
    
    # Sort by postcode and probability
    top_properties = top_properties.sort_values(['postcode_area', 'prediction_probability'], 
                                               ascending=[True, False])
    
    return top_properties

def save_predictions(predictions_df):
    """
    Save the predictions to a CSV file
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        
    Returns:
        str: Path to the saved predictions file
    """
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    predictions_file = reports_dir / f"top_properties_predictions_{today}.csv"
    
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to {predictions_file}")
    
    return str(predictions_file)

def generate_postcode_summary(predictions_df):
    """
    Generate a summary of predictions by postcode
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predictions
        
    Returns:
        pandas.DataFrame: Summary DataFrame
    """
    summary = predictions_df.groupby('postcode_area').agg(
        property_count=('property_id', 'count'),
        avg_probability=('prediction_probability', 'mean'),
        min_probability=('prediction_probability', 'min'),
        max_probability=('prediction_probability', 'max')
    ).reset_index()
    
    return summary

def main():
    """Main function to execute the prediction pipeline"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model")
            sys.exit(1)
        
        # Load prediction data
        df = load_prediction_data()
        if df.empty:
            logger.error("No prediction data available")
            sys.exit(1)
        
        # Prepare features
        X_pred, property_info = prepare_features(df)
        
        # Generate predictions
        predictions_df = generate_predictions(model, X_pred, property_info, config)
        
        if predictions_df.empty:
            logger.error("Failed to generate predictions")
            sys.exit(1)
        
        # Filter top properties by postcode
        top_properties = filter_top_properties_by_postcode(predictions_df, config)
        
        # Save predictions
        predictions_file = save_predictions(top_properties)
        
        # Generate summary
        summary = generate_postcode_summary(top_properties)
        
        # Save summary
        summary_file = project_root / 'reports' / f"prediction_summary_{datetime.now().strftime('%Y-%m-%d')}.csv"
        summary.to_csv(summary_file, index=False)
        logger.info(f"Saved prediction summary to {summary_file}")
        
        logger.info("Prediction pipeline completed successfully")
        logger.info(f"Total properties predicted: {len(top_properties)}")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 