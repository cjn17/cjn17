#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates visualizations and reports from the model predictions.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from datetime import datetime
import folium
from folium.plugins import MarkerCluster

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'generate_reports.log'),
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

def load_prediction_data():
    """
    Load the latest prediction data
    
    Returns:
        pandas.DataFrame: DataFrame containing prediction data
    """
    reports_dir = project_root / 'reports'
    
    # Find the latest prediction file
    files = list(reports_dir.glob("top_properties_predictions_*.csv"))
    if not files:
        logger.error("No prediction files found")
        return pd.DataFrame()
    
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Loading predictions from {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

def create_property_count_chart(df):
    """
    Create a bar chart showing the number of properties per postcode
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    logger.info("Creating property count chart")
    
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
    
    # Save the figure
    reports_dir = project_root / 'reports'
    chart_file = reports_dir / f"property_count_chart_{datetime.now().strftime('%Y-%m-%d')}.png"
    plt.savefig(chart_file, dpi=300)
    logger.info(f"Saved property count chart to {chart_file}")
    
    return plt.gcf()

def create_probability_distribution_chart(df):
    """
    Create a chart showing the distribution of prediction probabilities
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    logger.info("Creating probability distribution chart")
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    ax = sns.histplot(data=df, x='prediction_probability', hue='postcode_area', kde=True, alpha=0.6)
    
    plt.title('Distribution of Sale Prediction Probabilities by Postcode', fontsize=14)
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add vertical line for average probability
    plt.axvline(df['prediction_probability'].mean(), color='red', linestyle='--', 
                label=f'Average: {df["prediction_probability"].mean():.3f}')
    
    plt.legend(title='Postcode Area')
    plt.tight_layout()
    
    # Save the figure
    reports_dir = project_root / 'reports'
    chart_file = reports_dir / f"probability_distribution_chart_{datetime.now().strftime('%Y-%m-%d')}.png"
    plt.savefig(chart_file, dpi=300)
    logger.info(f"Saved probability distribution chart to {chart_file}")
    
    return plt.gcf()

def create_postcode_probability_boxplot(df):
    """
    Create a boxplot of prediction probabilities by postcode
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    logger.info("Creating postcode probability boxplot")
    
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    ax = sns.boxplot(data=df, x='postcode_area', y='prediction_probability', palette='viridis')
    
    # Add swarmplot for individual points
    sns.swarmplot(data=df, x='postcode_area', y='prediction_probability', color='black', alpha=0.5, size=4)
    
    plt.title('Prediction Probability Distribution by Postcode Area', fontsize=14)
    plt.xlabel('Postcode Area', fontsize=12)
    plt.ylabel('Prediction Probability', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    reports_dir = project_root / 'reports'
    chart_file = reports_dir / f"postcode_probability_boxplot_{datetime.now().strftime('%Y-%m-%d')}.png"
    plt.savefig(chart_file, dpi=300)
    logger.info(f"Saved postcode probability boxplot to {chart_file}")
    
    return plt.gcf()

def create_interactive_map(df):
    """
    Create an interactive map of predicted properties
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        folium.Map: Folium map object
    """
    logger.info("Creating interactive map")
    
    # For demonstration purposes, we need to generate fake coordinates
    # In a real implementation, you would use geocoded addresses
    
    # Create a map centered on East London
    m = folium.Map(location=[51.5500, 0.0700], zoom_start=11)
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each property
    for _, row in df.iterrows():
        # In a real implementation, get actual coordinates from geocoding
        # For now, generate random coordinates near the center
        lat = 51.5500 + np.random.normal(0, 0.02)
        lon = 0.0700 + np.random.normal(0, 0.02)
        
        # Calculate color based on probability (green for high, red for low)
        color = f'#{int(255 * (1 - row["prediction_probability"])):02x}{int(255 * row["prediction_probability"]):02x}00'
        
        # Create popup content
        popup_content = f"""
        <b>Address:</b> {row['address']}<br>
        <b>Postcode:</b> {row['postcode_area']}<br>
        <b>Probability:</b> {row['prediction_probability']:.3f}
        """
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row['address']} - {row['prediction_probability']:.3f}",
            icon=folium.Icon(color='green' if row['prediction_probability'] > 0.5 else 'red')
        ).add_to(marker_cluster)
    
    # Save the map
    reports_dir = project_root / 'reports'
    map_file = reports_dir / f"property_map_{datetime.now().strftime('%Y-%m-%d')}.html"
    m.save(map_file)
    logger.info(f"Saved interactive map to {map_file}")
    
    return m

def create_interactive_dashboard(df):
    """
    Create an interactive dashboard using Plotly
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        None
    """
    logger.info("Creating interactive dashboard")
    
    # Create a figure with subplots
    fig = go.Figure()
    
    # Add bar chart for property counts
    counts = df['postcode_area'].value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=counts.index,
        y=counts.values,
        name='Property Count',
        marker_color='skyblue'
    ))
    
    # Create buttons for different visualizations
    buttons = [
        {
            'label': "Property Counts",
            'method': "update",
            'args': [
                {"visible": [True, False, False]},
                {"title": "Number of Predicted Properties by Postcode Area"}
            ]
        }
    ]
    
    # Add histogram for probability distribution
    fig.add_trace(go.Histogram(
        x=df['prediction_probability'],
        name='Probability Distribution',
        marker_color='lightgreen',
        visible=False
    ))
    
    buttons.append({
        'label': "Probability Distribution",
        'method': "update",
        'args': [
            {"visible": [False, True, False]},
            {"title": "Distribution of Sale Prediction Probabilities"}
        ]
    })
    
    # Add box plot for postcode probabilities
    fig.add_trace(go.Box(
        x=df['postcode_area'],
        y=df['prediction_probability'],
        name='Probability by Postcode',
        marker_color='coral',
        visible=False
    ))
    
    buttons.append({
        'label': "Probability by Postcode",
        'method': "update",
        'args': [
            {"visible": [False, False, True]},
            {"title": "Prediction Probability Distribution by Postcode Area"}
        ]
    })
    
    # Update layout
    fig.update_layout(
        title="Number of Predicted Properties by Postcode Area",
        xaxis_title="Postcode Area",
        yaxis_title="Count / Probability",
        updatemenus=[{
            'active': 0,
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15
        }]
    )
    
    # Save the dashboard
    reports_dir = project_root / 'reports'
    dashboard_file = reports_dir / f"interactive_dashboard_{datetime.now().strftime('%Y-%m-%d')}.html"
    fig.write_html(dashboard_file)
    logger.info(f"Saved interactive dashboard to {dashboard_file}")

def create_summary_table(df):
    """
    Create a summary table of predictions by postcode
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        
    Returns:
        pandas.DataFrame: Summary DataFrame
    """
    logger.info("Creating summary table")
    
    # Group by postcode and calculate statistics
    summary = df.groupby('postcode_area').agg(
        property_count=('property_id', 'count'),
        avg_probability=('prediction_probability', 'mean'),
        min_probability=('prediction_probability', 'min'),
        max_probability=('prediction_probability', 'max'),
        median_probability=('prediction_probability', 'median')
    ).reset_index()
    
    # Format the probability columns
    for col in ['avg_probability', 'min_probability', 'max_probability', 'median_probability']:
        summary[col] = summary[col].map('{:.3f}'.format)
    
    # Save the summary table
    reports_dir = project_root / 'reports'
    summary_file = reports_dir / f"prediction_summary_{datetime.now().strftime('%Y-%m-%d')}.csv"
    summary.to_csv(summary_file, index=False)
    logger.info(f"Saved summary table to {summary_file}")
    
    return summary

def create_excel_report(df, summary_df):
    """
    Create a comprehensive Excel report with multiple sheets
    
    Args:
        df (pandas.DataFrame): DataFrame containing predictions
        summary_df (pandas.DataFrame): Summary DataFrame
        
    Returns:
        str: Path to Excel file
    """
    logger.info("Creating Excel report")
    
    reports_dir = project_root / 'reports'
    excel_file = reports_dir / f"property_predictions_report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Write summary sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write overall predictions sheet
        df_sorted = df.sort_values(['postcode_area', 'prediction_probability'], ascending=[True, False])
        df_sorted.to_excel(writer, sheet_name='All Predictions', index=False)
        
        # Write individual sheets for each postcode
        for postcode in df['postcode_area'].unique():
            postcode_df = df[df['postcode_area'] == postcode].sort_values('prediction_probability', ascending=False)
            postcode_df.to_excel(writer, sheet_name=f'Postcode {postcode}', index=False)
    
    logger.info(f"Saved Excel report to {excel_file}")
    
    return str(excel_file)

def main():
    """Main function to execute the report generation pipeline"""
    try:
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
        
        # Load configuration
        config = load_config()
        
        # Load prediction data
        df = load_prediction_data()
        
        if df.empty:
            logger.error("No prediction data available")
            sys.exit(1)
        
        # Create visualizations
        create_property_count_chart(df)
        create_probability_distribution_chart(df)
        create_postcode_probability_boxplot(df)
        
        # Create interactive visualizations
        create_interactive_map(df)
        create_interactive_dashboard(df)
        
        # Create summary table
        summary_df = create_summary_table(df)
        
        # Create Excel report
        excel_file = create_excel_report(df, summary_df)
        
        logger.info("Report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in report generation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 