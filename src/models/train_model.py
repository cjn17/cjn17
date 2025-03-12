#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script trains a predictive model to identify properties likely to be listed for sale.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'train_model.log'),
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

def load_training_data():
    """
    Load the processed data for model training
    
    Returns:
        pandas.DataFrame: DataFrame containing processed data
    """
    processed_dir = project_root / 'data' / 'processed'
    
    # Find the latest engineered features file
    files = list(processed_dir.glob("engineered_features_*.joblib"))
    if not files:
        logger.error("No engineered features files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading engineered features from {latest_file}")
    
    try:
        df = joblib.load(latest_file)
        return df
    except Exception as e:
        logger.error(f"Error loading engineered features: {e}")
        return pd.DataFrame()

def preprocess_data(df, config):
    """
    Preprocess the data for model training
    
    Args:
        df (pandas.DataFrame): DataFrame containing engineered features
        config (dict): Configuration dictionary
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Preprocessing data for model training")
    
    # Check if we have the target variable
    if 'sold_within_180_days' not in df.columns:
        logger.error("Target variable 'sold_within_180_days' not found in data")
        sys.exit(1)
    
    # Select relevant features for modeling
    feature_columns = [
        'years_since_last_sale', 'current_ownership_length', 
        'price_growth', 'annual_price_growth', 'price_growth_percentile',
        'transactions_last_year', 'transactions_last_6months', 'transactions_last_3months',
        'avg_price_last_year', 'median_price_last_year', 'price_volatility',
        'market_activity_trend', 'num_previous_sales', 'sales_frequency'
    ]
    
    # Add property type columns if they exist
    property_type_cols = [col for col in df.columns if col.startswith('property_type_')]
    feature_columns.extend(property_type_cols)
    
    # Add deprivation columns if they exist
    deprivation_cols = [col for col in df.columns if col.startswith('imd_')]
    feature_columns.extend(deprivation_cols)
    
    # Ensure all columns exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Handle missing values
    df = df.dropna(subset=['sold_within_180_days'])  # Drop rows with missing target
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
    
    # Split data into features and target
    X = df[feature_columns]
    y = df['sold_within_180_days']
    
    # Split data into training and testing sets
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    logger.info(f"Positive class ratio (training): {y_train.mean():.4f}")
    logger.info(f"Positive class ratio (testing): {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test

def create_model(config):
    """
    Create a machine learning model based on the configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        sklearn.pipeline.Pipeline: Model pipeline
    """
    model_type = config['model']['type']
    random_state = config['model']['random_state']
    
    logger.info(f"Creating {model_type} model")
    
    # Create the model pipeline with preprocessing
    pipeline_steps = [
        ('scaler', StandardScaler())
    ]
    
    # Add the appropriate model
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=random_state,
            class_weight='balanced',
            max_iter=1000
        )
        pipeline_steps.append(('model', model))
    
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight='balanced'
        )
        pipeline_steps.append(('model', model))
    
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        pipeline_steps.append(('model', model))
    
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)
    
    return Pipeline(pipeline_steps)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the model
    
    Args:
        model (sklearn.pipeline.Pipeline): Model pipeline
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training target
        y_test (pandas.Series): Testing target
        
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    logger.info("Training model")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate performance
    logger.info("Model evaluation")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Print classification report
    cr = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{cr}")
    
    # Perform cross-validation
    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    logger.info(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model

def save_model(model, config):
    """
    Save the trained model to disk
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        config (dict): Configuration dictionary
    """
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_type = config['model']['type']
    today = datetime.now().strftime('%Y-%m-%d')
    model_file = models_dir / f"{model_type}_model_{today}.joblib"
    
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")

def main():
    """Main function to execute the model training pipeline"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Load training data
        df = load_training_data()
        
        if df.empty:
            logger.error("No training data available")
            sys.exit(1)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, config)
        
        # Create model
        model = create_model(config)
        
        # Train and evaluate model
        trained_model = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Save model
        save_model(trained_model, config)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 