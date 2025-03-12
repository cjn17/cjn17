#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script fetches property transaction data from HM Land Registry
"""

import os
import sys
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'fetch_land_registry.log'),
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

def fetch_price_paid_data(config, start_date=None, end_date=None):
    """
    Fetch price paid data from HM Land Registry
    
    Args:
        config (dict): Configuration dictionary
        start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
        end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
    
    Returns:
        pandas.DataFrame: DataFrame containing price paid data
    """
    # Default dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching Land Registry data from {start_date} to {end_date}")
    
    base_url = config['data_sources']['land_registry']['url']
    postcodes = config['postcodes']
    
    # Create output directory if it doesn't exist
    output_dir = project_root / config['data_sources']['land_registry']['download_path']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Fetch data for each postcode
    for postcode in postcodes:
        logger.info(f"Fetching data for postcode: {postcode}")
        
        try:
            # For demonstration purposes, we're using a simplified API call
            # In a real implementation, this would need to be adapted to the actual Land Registry API
            params = {
                'postcode': postcode,
                'start_date': start_date,
                'end_date': end_date,
                'format': 'json'
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                # Process and transform the data (this will depend on the API structure)
                # This is a placeholder for the actual data processing
                postcode_df = pd.DataFrame(data.get('transactions', []))
                
                if not postcode_df.empty:
                    postcode_df['postcode_area'] = postcode
                    all_data.append(postcode_df)
                    
                    # Save individual postcode data
                    output_file = output_dir / f"{postcode}_transactions_{start_date}_to_{end_date}.csv"
                    postcode_df.to_csv(output_file, index=False)
                    logger.info(f"Saved data for {postcode} to {output_file}")
            else:
                logger.error(f"Failed to fetch data for {postcode}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error processing postcode {postcode}: {e}")
    
    if all_data:
        # Combine all postcode data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined data
        combined_output_file = output_dir / f"all_transactions_{start_date}_to_{end_date}.csv"
        combined_df.to_csv(combined_output_file, index=False)
        logger.info(f"Saved combined data to {combined_output_file}")
        
        return combined_df
    else:
        logger.warning("No data was fetched.")
        return pd.DataFrame()

def main():
    """Main function to execute the script"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Fetch data for the last 5 years to build historical dataset
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        df = fetch_price_paid_data(config, start_date, end_date)
        
        logger.info(f"Successfully fetched Land Registry data: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 