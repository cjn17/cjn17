# Property Sale Predictor

A machine learning system to predict which residential properties in East London are most likely to be listed for sale within 180 days.

## Project Objective

- Predict residential properties most likely to list for sale within 180 days in specific East London postcodes:
  - RM6, RM8, RM9, RM10, IG1, IG2, IG3, IG4, IG5, IG6, IG11
- Monthly analysis of approximately 150,000 properties
- Output the top 50 most likely-to-sell properties per postcode each month

## Project Structure

- `data/`: Directory for raw and processed data
  - `raw/`: Raw data from various sources
  - `processed/`: Cleaned and processed datasets
- `notebooks/`: Jupyter notebooks for exploration and model development
- `src/`: Source code
  - `data/`: Scripts for data acquisition and processing
  - `features/`: Scripts for feature engineering
  - `models/`: Model training and prediction code
  - `visualization/`: Code for generating visualizations and reports
- `config/`: Configuration files for the project
- `reports/`: Generated reports and visualizations
- `tests/`: Unit and integration tests

## Data Sources

- HM Land Registry Price Paid Data
- Current market listings (Rightmove)
- ONS Census data
- Index of Multiple Deprivation
- Optional: Experian Mosaic or similar segmentation data

## Feature Engineering

- Length of ownership
- Estimated current value vs. original purchase price
- Local market conditions
- Property attributes
- Area-level socio-economic proxies
- Optional premium data features

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/property-sale-predictor.git
cd property-sale-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Update data sources:
```bash
python src/data/update_data.py
```

2. Run the prediction pipeline:
```bash
python src/models/predict.py
```

3. Generate reports:
```bash
python src/visualization/generate_reports.py
```

## Dashboard

The prediction results can be visualized in the Power BI dashboard (see `reports/dashboard.pbix`).

## Automation

The prediction pipeline is scheduled to run monthly using cron jobs or a similar scheduler.

## License

[Specify License] 