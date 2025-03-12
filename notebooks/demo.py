#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Property Sale Prediction System Demo

This script demonstrates how to use the property sale prediction system to identify
residential properties most likely to be listed for sale within 180 days in specific
East London postcodes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import joblib
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Set up visualization defaults
plt.style.use('ggplot')
sns.set_theme()

def load_config():
    """Load the configuration file"""
    try:
        with open(project_root / 'config' / 'config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def display_config(config):
    """Display key configuration settings"""
    print("=== Project Configuration ===")
    
    # Display target postcodes
    print("\nTarget Postcodes:")
    for postcode in config['postcodes']:
        print(f"- {postcode}")
    
    # Display model configuration
    print("\nModel Configuration:")
    for key, value in config['model'].items():
        print(f"- {key}: {value}")
    
    # Display feature engineering settings
    print("\nFeature Engineering:")
    for key, value in config['features'].items():
        print(f"- {key}: {value['enabled']}")
    
    # Display automation settings
    print("\nAutomation Settings:")
    for key, value in config['automation'].items():
        print(f"- {key}: {value}")

def load_latest_predictions():
    """Load the latest prediction results"""
    reports_dir = project_root / 'reports'
    files = list(reports_dir.glob("top_properties_predictions_*.csv"))
    if not files:
        print("No prediction files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading predictions from {latest_file}")
    
    try:
        return pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()

def display_predictions(df):
    """Display prediction summary"""
    if df.empty:
        print("No predictions available. Run the pipeline first to generate predictions.")
        return
    
    print(f"\n=== Prediction Summary ===")
    print(f"Total predictions: {len(df)}")
    
    # Display summary by postcode
    print("\nProperties by Postcode:")
    postcode_counts = df['postcode_area'].value_counts().sort_index()
    for postcode, count in postcode_counts.items():
        print(f"- {postcode}: {count} properties")
    
    # Display top 10 properties
    print("\nTop 10 Properties Most Likely to Sell:")
    top_10 = df.sort_values('prediction_probability', ascending=False).head(10)
    for idx, row in top_10.iterrows():
        print(f"- {row['address']} ({row['postcode_area']}): {row['prediction_probability']:.3f} probability")

def visualize_predictions(df):
    """Create visualizations of the predictions"""
    if df.empty:
        print("No predictions available for visualization.")
        return
    
    print("\n=== Creating Visualizations ===")
    
    # Create output directory for visualizations
    vis_dir = project_root / 'reports' / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Bar chart of property counts by postcode
    print("Creating property count chart...")
    plt.figure(figsize=(10, 6))
    counts = df['postcode_area'].value_counts().sort_index()
    ax = counts.plot(kind='bar', color='skyblue')
    
    plt.title('Number of Predicted Properties by Postcode Area', fontsize=14)
    plt.xlabel('Postcode Area', fontsize=12)
    plt.ylabel('Number of Properties', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(vis_dir / f"property_count_chart_{timestamp}.png", dpi=300)
    plt.close()
    
    # 2. Histogram of prediction probabilities
    print("Creating probability distribution chart...")
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    ax = sns.histplot(data=df, x='prediction_probability', kde=True, alpha=0.6)
    
    plt.title('Distribution of Sale Prediction Probabilities', fontsize=14)
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add vertical line for average probability
    plt.axvline(df['prediction_probability'].mean(), color='red', linestyle='--', 
                label=f'Average: {df["prediction_probability"].mean():.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / f"probability_distribution_chart_{timestamp}.png", dpi=300)
    plt.close()
    
    # 3. Boxplot of prediction probabilities by postcode
    print("Creating postcode probability boxplot...")
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    ax = sns.boxplot(data=df, x='postcode_area', y='prediction_probability', palette='viridis')
    
    plt.title('Prediction Probability Distribution by Postcode Area', fontsize=14)
    plt.xlabel('Postcode Area', fontsize=12)
    plt.ylabel('Prediction Probability', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(vis_dir / f"postcode_probability_boxplot_{timestamp}.png", dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

def load_latest_model():
    """Load the latest trained model"""
    models_dir = project_root / 'models'
    files = list(models_dir.glob("*_model_*.joblib"))
    if not files:
        print("No model files found")
        return None
    
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading model from {latest_file}")
    
    try:
        return joblib.load(latest_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def display_feature_importance(model):
    """Display feature importance from the model"""
    if model is None:
        print("No model available. Train a model first to view feature importances.")
        return
    
    print("\n=== Feature Importance ===")
    
    try:
        # Get the actual model from the pipeline
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            estimator = model.named_steps['model']
            
            # Check if model has feature_importances_ attribute (Random Forest or XGBoost)
            if hasattr(estimator, 'feature_importances_'):
                # Load the latest model data to get feature names
                processed_dir = project_root / 'data' / 'processed'
                files = list(processed_dir.glob("model_data_*.csv"))
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    model_data = pd.read_csv(latest_file)
                    
                    # Get feature names (excluding identifiers)
                    feature_names = [col for col in model_data.columns 
                                     if col not in ['property_id', 'address', 'postcode_area']]
                    
                    # Create DataFrame with feature importances
                    if len(feature_names) == len(estimator.feature_importances_):
                        importances = pd.DataFrame({
                            'feature': feature_names,
                            'importance': estimator.feature_importances_
                        })
                        
                        # Sort by importance
                        importances = importances.sort_values('importance', ascending=False)
                        
                        # Display top 15 importances
                        print("Top 15 Most Important Features:")
                        for idx, row in importances.head(15).iterrows():
                            print(f"- {row['feature']}: {row['importance']:.4f}")
                    else:
                        print("Mismatch between feature names and importance values")
                else:
                    print("No model data files found")
            # For logistic regression models
            elif hasattr(estimator, 'coef_'):
                print("Logistic Regression model detected - coefficients could be extracted")
            else:
                print("Model does not have feature importances or coefficients")
        else:
            print("Model is not a pipeline with a 'model' step")
    except Exception as e:
        print(f"Error extracting feature importances: {e}")

def show_how_to_run_pipeline():
    """Display instructions for running the pipeline"""
    print("\n=== How to Run the Pipeline ===")
    print("To run the complete pipeline, execute the following command from the project root:")
    print("  python src/main.py")
    print("\nOptions:")
    print("  --data-only: Only run data acquisition and feature engineering")
    print("  --skip-training: Skip model training (use existing model)")
    print("  --schedule: Run in scheduled mode based on configuration")
    print("\nExample for scheduled execution:")
    print("  python src/main.py --schedule")

def main():
    """Main function demonstrating the system"""
    print("\n" + "="*50)
    print("PROPERTY SALE PREDICTION SYSTEM DEMONSTRATION")
    print("="*50)
    
    # Load and display configuration
    config = load_config()
    display_config(config)
    
    # Load and display predictions
    predictions_df = load_latest_predictions()
    display_predictions(predictions_df)
    
    # Visualize predictions if available
    if not predictions_df.empty:
        visualize_predictions(predictions_df)
    
    # Display feature importance
    model = load_latest_model()
    if model is not None:
        display_feature_importance(model)
    
    # Show how to run the pipeline
    show_how_to_run_pipeline()
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)

if __name__ == '__main__':
    main() 