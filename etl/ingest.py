import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd

# Ensure working directory is set to project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/staging', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/ingest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ingest_data(force=False):
    # Locate raw Excel file (either in data/raw or root)
    raw_path = 'data/raw/Online Retail.xlsx'
    if not os.path.exists(raw_path):
        if os.path.exists('Online Retail.xlsx'):
            raw_path = 'Online Retail.xlsx'
        else:
            raise FileNotFoundError(f"Raw dataset not found at 'data/raw/Online Retail.xlsx' or 'Online Retail.xlsx'.")
            
    staging_path = 'data/staging/stg_online_retail.csv'
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source': 'local_file',
        'filename': os.path.basename(raw_path),
        'status': 'failed',
        'row_count': 0,
        'column_count': 0,
        'error': None
    }
    
    try:
        logging.info(f"Starting ingestion from {raw_path}")
        
        # Check if staging CSV exists and is newer than raw file, unless forced
        staging_exists = os.path.exists(staging_path)
        raw_mtime = os.path.getmtime(raw_path) if os.path.exists(raw_path) else 0
        stg_mtime = os.path.getmtime(staging_path) if staging_exists else 0
        
        if staging_exists and not force and stg_mtime >= raw_mtime:
            logging.info(f"Staging file {staging_path} is up-to-date. Reading CSV for fast ingestion.")
            df = pd.read_csv(staging_path)
        else:
            logging.info(f"Reading raw Excel file from {raw_path}...")
            print(f"Reading raw Excel file ({raw_path})... this may take a few seconds.")
            df = pd.read_excel(raw_path)
            df.to_csv(staging_path, index=False)
            logging.info(f"Staging CSV created at {staging_path}.")
        
        metadata['row_count'] = len(df)
        metadata['column_count'] = len(df.columns)
        metadata['status'] = 'success'
        logging.info(f"Successfully ingested {len(df)} rows. Columns: {list(df.columns)}")
        print(f"Ingestion success. Rows: {metadata['row_count']}, Columns: {metadata['column_count']}")
        return df
        
    except Exception as e:
        metadata['error'] = str(e)
        logging.error(f"Ingestion failed: {str(e)}")
        print(f"Ingestion failed: {e}")
        raise
    
    finally:
        with open('logs/ingest_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == '__main__':
    ingest_data()

