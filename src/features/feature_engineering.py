#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs feature engineering on the raw data to prepare it for modeling.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import joblib

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'feature_engineering.log'),
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

def load_land_registry_data(config):
    """
    Load the Land Registry price paid data
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        pandas.DataFrame: DataFrame containing price paid data
    """
    land_registry_dir = project_root / config['data_sources']['land_registry']['download_path']
    
    # Find the latest combined file
    files = list(land_registry_dir.glob("all_transactions_*.csv"))
    if not files:
        logger.error("No Land Registry data files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading Land Registry data from {latest_file}")
    
    return pd.read_csv(latest_file)

def load_rightmove_data(config):
    """
    Load the Rightmove current listings data
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        pandas.DataFrame: DataFrame containing current listings
    """
    rightmove_dir = project_root / config['data_sources']['rightmove']['download_path']
    
    # Find the latest combined file
    files = list(rightmove_dir.glob("all_listings_*.csv"))
    if not files:
        logger.error("No Rightmove data files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading Rightmove data from {latest_file}")
    
    return pd.read_csv(latest_file)

def load_deprivation_data(config):
    """
    Load the Index of Multiple Deprivation data
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        pandas.DataFrame: DataFrame containing deprivation data
    """
    deprivation_file = project_root / config['data_sources']['deprivation']['download_path'] / "imd2019.csv"
    
    if not deprivation_file.exists():
        logger.warning(f"Deprivation data file not found at {deprivation_file}")
        return pd.DataFrame()
    
    logger.info(f"Loading deprivation data from {deprivation_file}")
    return pd.read_csv(deprivation_file)

def compute_ownership_length(df):
    """
    Compute the length of ownership for each property
    
    Args:
        df (pandas.DataFrame): DataFrame containing property transactions
        
    Returns:
        pandas.DataFrame: DataFrame with ownership length feature added
    """
    logger.info("Computing ownership length")
    
    # Ensure the date column is in datetime format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Group by property and sort by date
    df = df.sort_values(['property_id', 'transaction_date'])
    
    # Calculate time since last sale
    df['last_sale_date'] = df.groupby('property_id')['transaction_date'].shift(1)
    df['years_since_last_sale'] = (df['transaction_date'] - df['last_sale_date']).dt.days / 365.25
    
    # For first sales, set to NaN
    df.loc[df['last_sale_date'].isna(), 'years_since_last_sale'] = np.nan
    
    # Calculate the current ownership length (time since last transaction)
    today = datetime.now()
    df['current_ownership_length'] = (today - df.groupby('property_id')['transaction_date'].transform('max')).dt.days / 365.25
    
    return df

def compute_price_growth(df):
    """
    Compute the price growth for each property
    
    Args:
        df (pandas.DataFrame): DataFrame containing property transactions
        
    Returns:
        pandas.DataFrame: DataFrame with price growth features added
    """
    logger.info("Computing price growth")
    
    # Ensure price and date columns are in the correct format
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Group by property and sort by date
    df = df.sort_values(['property_id', 'transaction_date'])
    
    # Calculate previous price
    df['previous_price'] = df.groupby('property_id')['price'].shift(1)
    
    # Calculate price growth
    df['price_growth'] = (df['price'] - df['previous_price']) / df['previous_price']
    df['annual_price_growth'] = df['price_growth'] / (df['years_since_last_sale'] + 0.0001)  # Avoid division by zero
    
    # Calculate price growth percentile within postcode
    df['price_growth_percentile'] = df.groupby('postcode_area')['price_growth'].transform(
        lambda x: x.rank(pct=True)
    )
    
    return df

def compute_local_market_features(df):
    """
    Compute local market condition features
    
    Args:
        df (pandas.DataFrame): DataFrame containing property transactions
        
    Returns:
        pandas.DataFrame: DataFrame with local market features added
    """
    logger.info("Computing local market features")
    
    # Ensure date column is in datetime format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Calculate recent transaction counts by postcode
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    six_months_ago = today - timedelta(days=180)
    three_months_ago = today - timedelta(days=90)
    
    # Create flags for recent transactions
    df['is_last_year'] = df['transaction_date'] >= one_year_ago
    df['is_last_6months'] = df['transaction_date'] >= six_months_ago
    df['is_last_3months'] = df['transaction_date'] >= three_months_ago
    
    # Aggregate by postcode for recent transaction counts
    postcode_stats = df.groupby('postcode_area').agg(
        transactions_last_year=('is_last_year', 'sum'),
        transactions_last_6months=('is_last_6months', 'sum'),
        transactions_last_3months=('is_last_3months', 'sum'),
        avg_price_last_year=pd.NamedAgg(column='price', aggfunc=lambda x: x[df['is_last_year']].mean()),
        median_price_last_year=pd.NamedAgg(column='price', aggfunc=lambda x: x[df['is_last_year']].median()),
        price_volatility=pd.NamedAgg(column='price', aggfunc=lambda x: x[df['is_last_year']].std() / x[df['is_last_year']].mean())
    )
    
    # Merge postcode stats back to the main dataframe
    df = df.merge(postcode_stats, on='postcode_area', how='left')
    
    # Calculate market activity metrics
    df['market_activity_trend'] = df['transactions_last_3months'] / (df['transactions_last_6months'] / 2 + 0.001)
    
    return df

def compute_property_attributes(df):
    """
    Extract and compute property attribute features
    
    Args:
        df (pandas.DataFrame): DataFrame containing property transactions
        
    Returns:
        pandas.DataFrame: DataFrame with property attribute features added
    """
    logger.info("Computing property attribute features")
    
    # Extract property type from property_type column if available
    if 'property_type' in df.columns:
        # One-hot encode property type
        property_type_dummies = pd.get_dummies(df['property_type'], prefix='property_type')
        df = pd.concat([df, property_type_dummies], axis=1)
    
    # Calculate number of previous sales for each property
    df['num_previous_sales'] = df.groupby('property_id').cumcount()
    
    # Calculate sale frequency (sales per year since first record)
    df['first_sale_date'] = df.groupby('property_id')['transaction_date'].transform('min')
    df['years_since_first_sale'] = (df['transaction_date'] - df['first_sale_date']).dt.days / 365.25
    df['sales_frequency'] = df['num_previous_sales'] / (df['years_since_first_sale'] + 0.001)  # Avoid division by zero
    
    return df

def merge_deprivation_data(df, deprivation_df):
    """
    Merge deprivation data with the main dataframe
    
    Args:
        df (pandas.DataFrame): Main DataFrame with property transactions
        deprivation_df (pandas.DataFrame): DataFrame with deprivation data
        
    Returns:
        pandas.DataFrame: Merged DataFrame with deprivation features
    """
    if deprivation_df.empty:
        logger.warning("No deprivation data available for merging")
        return df
    
    logger.info("Merging deprivation data")
    
    # Perform the merge based on postcode
    # This is a simplified example - in reality you would need to map to LSOA codes
    merged_df = df.merge(deprivation_df, on='postcode', how='left')
    
    return merged_df

def identify_already_listed_properties(transactions_df, listings_df):
    """
    Identify properties that are already listed on the market
    
    Args:
        transactions_df (pandas.DataFrame): DataFrame with property transactions
        listings_df (pandas.DataFrame): DataFrame with current property listings
        
    Returns:
        pandas.DataFrame: DataFrame with flag for already listed properties
    """
    if listings_df.empty:
        logger.warning("No current listings data available")
        return transactions_df
    
    logger.info("Identifying already listed properties")
    
    # Create a set of addresses that are currently listed
    current_listings = set(listings_df['address'].str.lower())
    
    # Flag properties that are already listed
    transactions_df['already_listed'] = transactions_df['address'].str.lower().isin(current_listings)
    
    return transactions_df

def create_target_variable(df):
    """
    Create the target variable: property sold within 180 days
    
    Args:
        df (pandas.DataFrame): DataFrame with property transactions
        
    Returns:
        pandas.DataFrame: DataFrame with target variable added
    """
    logger.info("Creating target variable")
    
    # Sort by property and date
    df = df.sort_values(['property_id', 'transaction_date'])
    
    # Calculate time to next sale for each property
    df['next_sale_date'] = df.groupby('property_id')['transaction_date'].shift(-1)
    df['days_to_next_sale'] = (df['next_sale_date'] - df['transaction_date']).dt.days
    
    # Create target variable - sold within 180 days
    df['sold_within_180_days'] = (df['days_to_next_sale'] <= 180) & (df['days_to_next_sale'] > 0)
    
    # For properties without a next sale date, set to False
    df['sold_within_180_days'] = df['sold_within_180_days'].fillna(False)
    
    return df

def prepare_model_data(df):
    """
    Prepare the data for modeling by handling missing values and encoding categorical variables
    
    Args:
        df (pandas.DataFrame): DataFrame with all features
        
    Returns:
        pandas.DataFrame: DataFrame ready for modeling
    """
    logger.info("Preparing data for modeling")
    
    # Identify the latest record for each property
    df = df.sort_values(['property_id', 'transaction_date'])
    latest_records = df.groupby('property_id').tail(1).copy()
    
    # Filter out properties that are already listed
    if 'already_listed' in latest_records.columns:
        latest_records = latest_records[~latest_records['already_listed']]
    
    # Select relevant features
    model_features = [
        'years_since_last_sale', 'current_ownership_length', 
        'price_growth', 'annual_price_growth', 'price_growth_percentile',
        'transactions_last_year', 'transactions_last_6months', 'transactions_last_3months',
        'avg_price_last_year', 'median_price_last_year', 'price_volatility',
        'market_activity_trend', 'num_previous_sales', 'sales_frequency'
    ]
    
    # Add property type columns if they exist
    property_type_cols = [col for col in latest_records.columns if col.startswith('property_type_')]
    model_features.extend(property_type_cols)
    
    # Add deprivation columns if they exist
    deprivation_cols = [col for col in latest_records.columns if col.startswith('imd_')]
    model_features.extend(deprivation_cols)
    
    # Ensure all required columns exist
    model_features = [col for col in model_features if col in latest_records.columns]
    
    # Select features and target
    X = latest_records[model_features]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Include property identifier and address for output
    X['property_id'] = latest_records['property_id']
    X['address'] = latest_records['address']
    X['postcode_area'] = latest_records['postcode_area']
    
    return X

def main():
    """Main function to execute the feature engineering pipeline"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Load raw data
        land_registry_df = load_land_registry_data(config)
        rightmove_df = load_rightmove_data(config)
        deprivation_df = load_deprivation_data(config)
        
        if land_registry_df.empty:
            logger.error("No Land Registry data available")
            sys.exit(1)
        
        # Perform feature engineering
        df = land_registry_df.copy()
        
        # Only perform feature engineering if enabled in config
        if config['features']['ownership_length']['enabled']:
            df = compute_ownership_length(df)
        
        if config['features']['price_growth']['enabled']:
            df = compute_price_growth(df)
        
        if config['features']['local_market']['enabled']:
            df = compute_local_market_features(df)
        
        if config['features']['property_attributes']['enabled']:
            df = compute_property_attributes(df)
        
        if config['features']['socioeconomic']['enabled'] and not deprivation_df.empty:
            df = merge_deprivation_data(df, deprivation_df)
        
        # Identify properties already on the market
        if not rightmove_df.empty:
            df = identify_already_listed_properties(df, rightmove_df)
        
        # Create target variable for training
        df = create_target_variable(df)
        
        # Prepare data for modeling
        model_data = prepare_model_data(df)
        
        # Save processed data
        processed_dir = project_root / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        output_file = processed_dir / f"model_data_{today}.csv"
        model_data.to_csv(output_file, index=False)
        
        # Save engineered features for future use
        features_file = processed_dir / f"engineered_features_{today}.joblib"
        joblib.dump(df, features_file)
        
        logger.info(f"Feature engineering completed. Saved model data to {output_file}")
        logger.info(f"Created {len(model_data)} records for modeling")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 