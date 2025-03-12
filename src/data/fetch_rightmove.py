#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script fetches current property listings from Rightmove
to exclude properties already on the market.
"""

import os
import sys
import yaml
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import time
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'fetch_rightmove.log'),
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

def setup_selenium():
    """Set up the Selenium WebDriver with Chrome"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"Error setting up Selenium: {e}")
        raise

def scrape_rightmove(config):
    """
    Scrape property listings from Rightmove
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        pandas.DataFrame: DataFrame containing property listings
    """
    base_url = config['data_sources']['rightmove']['base_url']
    postcodes = config['postcodes']
    
    # Create output directory if it doesn't exist
    output_dir = project_root / config['data_sources']['rightmove']['download_path']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    all_listings = []
    
    driver = setup_selenium()
    
    for postcode in postcodes:
        logger.info(f"Fetching listings for postcode: {postcode}")
        
        listings = []
        page = 0
        max_pages = 50  # Safety limit
        
        try:
            while page < max_pages:
                # Construct the URL for the current page
                if page == 0:
                    url = f"{base_url}?searchType=SALE&locationIdentifier=OUTCODE%^{postcode}&radius=0.5"
                else:
                    url = f"{base_url}?searchType=SALE&locationIdentifier=OUTCODE%^{postcode}&radius=0.5&index={page * 24}"
                
                # Add a random delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
                
                logger.info(f"Fetching page {page + 1} for {postcode}")
                driver.get(url)
                
                try:
                    # Wait for the property cards to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "propertyCard"))
                    )
                    
                    # Parse the page
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    # Find all property cards
                    property_cards = soup.find_all("div", class_="propertyCard")
                    
                    if not property_cards:
                        logger.info(f"No more listings found for {postcode} on page {page + 1}")
                        break
                    
                    # Extract data from each property card
                    for card in property_cards:
                        try:
                            property_data = {}
                            
                            # Extract address
                            address_elem = card.find("address", class_="propertyCard-address")
                            if address_elem:
                                property_data['address'] = address_elem.text.strip()
                            
                            # Extract price
                            price_elem = card.find("div", class_="propertyCard-priceValue")
                            if price_elem:
                                property_data['price'] = price_elem.text.strip()
                            
                            # Extract property type
                            type_elem = card.find("h2", class_="propertyCard-title")
                            if type_elem:
                                property_data['property_type'] = type_elem.text.strip()
                            
                            # Extract listing URL
                            url_elem = card.find("a", class_="propertyCard-link")
                            if url_elem and 'href' in url_elem.attrs:
                                property_data['url'] = "https://www.rightmove.co.uk" + url_elem['href']
                                
                                # Extract property ID from URL
                                property_id = url_elem['href'].split('/')[-1].split('?')[0]
                                property_data['property_id'] = property_id
                            
                            property_data['postcode_area'] = postcode
                            property_data['fetch_date'] = today
                            
                            listings.append(property_data)
                        except Exception as e:
                            logger.error(f"Error extracting property data: {e}")
                    
                    page += 1
                    
                except Exception as e:
                    logger.error(f"Error parsing page {page + 1} for {postcode}: {e}")
                    break
            
            # Create DataFrame for this postcode
            if listings:
                postcode_df = pd.DataFrame(listings)
                all_listings.append(postcode_df)
                
                # Save individual postcode data
                output_file = output_dir / f"{postcode}_listings_{today}.csv"
                postcode_df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(postcode_df)} listings for {postcode} to {output_file}")
            else:
                logger.warning(f"No listings found for {postcode}")
                
        except Exception as e:
            logger.error(f"Error processing postcode {postcode}: {e}")
    
    # Close the WebDriver
    driver.quit()
    
    if all_listings:
        # Combine all postcode data
        combined_df = pd.concat(all_listings, ignore_index=True)
        
        # Save combined data
        combined_output_file = output_dir / f"all_listings_{today}.csv"
        combined_df.to_csv(combined_output_file, index=False)
        logger.info(f"Saved combined {len(combined_df)} listings to {combined_output_file}")
        
        return combined_df
    else:
        logger.warning("No listings were fetched.")
        return pd.DataFrame()

def main():
    """Main function to execute the script"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Scrape Rightmove listings
        df = scrape_rightmove(config)
        
        logger.info(f"Successfully fetched Rightmove listings: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 