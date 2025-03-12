#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for the property sale prediction pipeline.
This orchestrates the entire process from data acquisition to report generation.
"""

import os
import sys
import yaml
import argparse
import logging
import time
import schedule
from datetime import datetime
from pathlib import Path
import importlib.util
import traceback

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'main.log'),
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

def run_module(module_path, module_name):
    """
    Run a Python module by importing and executing its main function
    
    Args:
        module_path (str): Path to the module file
        module_name (str): Name of the module for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Running {module_name}...")
    
    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call the main function
        if hasattr(module, 'main'):
            module.main()
            logger.info(f"{module_name} completed successfully")
            return True
        else:
            logger.error(f"{module_name} does not have a main function")
            return False
            
    except Exception as e:
        logger.error(f"Error running {module_name}: {e}")
        logger.error(traceback.format_exc())
        return False

def run_pipeline(data_only=False, skip_training=False):
    """
    Run the complete prediction pipeline
    
    Args:
        data_only (bool): If True, only run data acquisition and feature engineering
        skip_training (bool): If True, skip the model training step
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting property sale prediction pipeline")
    
    # Create necessary directories
    (project_root / 'logs').mkdir(exist_ok=True)
    (project_root / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    (project_root / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (project_root / 'models').mkdir(exist_ok=True)
    (project_root / 'reports').mkdir(exist_ok=True)
    
    # Step 1: Fetch Land Registry data
    land_registry_success = run_module(
        project_root / 'src' / 'data' / 'fetch_land_registry.py',
        'fetch_land_registry'
    )
    
    if not land_registry_success:
        logger.error("Land Registry data acquisition failed. Aborting pipeline.")
        return False
    
    # Step 2: Fetch Rightmove data
    rightmove_success = run_module(
        project_root / 'src' / 'data' / 'fetch_rightmove.py',
        'fetch_rightmove'
    )
    
    if not rightmove_success:
        logger.warning("Rightmove data acquisition failed. Continuing without current listings data.")
    
    # Step 3: Perform feature engineering
    feature_engineering_success = run_module(
        project_root / 'src' / 'features' / 'feature_engineering.py',
        'feature_engineering'
    )
    
    if not feature_engineering_success:
        logger.error("Feature engineering failed. Aborting pipeline.")
        return False
    
    # If data_only flag is set, stop here
    if data_only:
        logger.info("Data acquisition and feature engineering completed successfully. Skipping model training and prediction as requested.")
        return True
    
    # Step 4: Train model (if not skipped)
    if not skip_training:
        training_success = run_module(
            project_root / 'src' / 'models' / 'train_model.py',
            'train_model'
        )
        
        if not training_success:
            logger.error("Model training failed. Aborting pipeline.")
            return False
    else:
        logger.info("Skipping model training as requested.")
    
    # Step 5: Generate predictions
    prediction_success = run_module(
        project_root / 'src' / 'models' / 'predict.py',
        'predict'
    )
    
    if not prediction_success:
        logger.error("Prediction generation failed. Aborting pipeline.")
        return False
    
    # Step 6: Generate reports and visualizations
    reporting_success = run_module(
        project_root / 'src' / 'visualization' / 'generate_reports.py',
        'generate_reports'
    )
    
    if not reporting_success:
        logger.warning("Report generation failed. Pipeline completed with warnings.")
    
    logger.info("Property sale prediction pipeline completed successfully")
    return True

def schedule_pipeline(config):
    """
    Schedule the pipeline to run automatically based on configuration
    
    Args:
        config (dict): Configuration dictionary
    """
    schedule_type = config['automation']['schedule']
    
    if schedule_type == 'monthly':
        day = config['automation']['day_of_month']
        logger.info(f"Scheduling pipeline to run monthly on day {day}")
        
        # Schedule monthly on specific day
        schedule.every().month.on(day).at("00:00").do(run_pipeline)
        
    elif schedule_type == 'weekly':
        logger.info("Scheduling pipeline to run weekly on Monday")
        schedule.every().monday.at("00:00").do(run_pipeline)
        
    elif schedule_type == 'daily':
        logger.info("Scheduling pipeline to run daily at midnight")
        schedule.every().day.at("00:00").do(run_pipeline)
    
    else:
        logger.error(f"Unsupported schedule type: {schedule_type}")
        return
    
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Property Sale Prediction Pipeline')
    
    parser.add_argument('--data-only', action='store_true', 
                        help='Only run data acquisition and feature engineering')
    
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training step')
    
    parser.add_argument('--schedule', action='store_true',
                        help='Run in scheduled mode based on configuration')
    
    return parser.parse_args()

def main():
    """Main entry point for the script"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config()
        
        # Run in scheduled mode or one-time mode
        if args.schedule:
            schedule_pipeline(config)
        else:
            # Run pipeline once
            run_pipeline(data_only=args.data_only, skip_training=args.skip_training)
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 