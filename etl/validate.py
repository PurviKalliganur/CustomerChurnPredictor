import os
import sys
import json
import logging
import pandas as pd

# Ensure working directory is project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/rejected', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

logging.basicConfig(
    filename='logs/validate.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_data():
    staging_path = 'data/staging/stg_online_retail.csv'
    processed_path = 'data/processed/val_online_retail.csv'
    
    logging.info("Starting data validation...")
    if not os.path.exists(staging_path):
        raise FileNotFoundError(f"Staging file not found at {staging_path}. Please run ingest step first.")
        
    df = pd.read_csv(staging_path)
    initial_count = len(df)
    
    # 1. Duplicates
    duplicates = df[df.duplicated(keep='first')]
    duplicate_count = len(duplicates)
    if duplicate_count > 0:
        duplicates.to_csv('data/rejected/duplicates.csv', index=False)
        df = df.drop_duplicates(keep='first')
        
    # 2. Missing Critical Fields (CustomerID, Description, InvoiceNo, StockCode)
    missing_mask = df[['CustomerID', 'Description', 'InvoiceNo', 'StockCode']].isnull().any(axis=1)
    missing_records = df[missing_mask]
    missing_count = len(missing_records)
    if missing_count > 0:
        missing_records.to_csv('data/rejected/missing_fields.csv', index=False)
        df = df[~missing_mask]
        
    # 3. Cancelled Invoices (InvoiceNo starts with 'C')
    cancelled_mask = df['InvoiceNo'].astype(str).str.startswith('C')
    cancelled_records = df[cancelled_mask]
    cancelled_count = len(cancelled_records)
    if cancelled_count > 0:
        cancelled_records.to_csv('data/rejected/cancelled_invoices.csv', index=False)
        df = df[~cancelled_mask]
        
    # 4. Invalid Quantity (Quantity <= 0)
    invalid_qty_mask = df['Quantity'] <= 0
    invalid_qty_records = df[invalid_qty_mask]
    invalid_qty_count = len(invalid_qty_records)
    if invalid_qty_count > 0:
        invalid_qty_records.to_csv('data/rejected/invalid_quantity.csv', index=False)
        df = df[~invalid_qty_mask]
        
    # 5. Invalid UnitPrice (UnitPrice <= 0)
    invalid_price_mask = df['UnitPrice'] <= 0
    invalid_price_records = df[invalid_price_mask]
    invalid_price_count = len(invalid_price_records)
    if invalid_price_count > 0:
        invalid_price_records.to_csv('data/rejected/invalid_price.csv', index=False)
        df = df[~invalid_price_mask]
        
    # Ensure CustomerID is integer in valid records
    df['CustomerID'] = df['CustomerID'].astype(int)
    valid_count = len(df)
    
    # Save valid data
    df.to_csv(processed_path, index=False)
    logging.info(f"Validation complete. Valid rows: {valid_count}")
    
    # Generate Data Quality Report
    dq_report = {
        'total_rows_input': initial_count,
        'valid_rows': valid_count,
        'total_rejected_rows': initial_count - valid_count,
        'rejection_reasons': {
            'duplicates': int(duplicate_count),
            'missing_critical_fields': int(missing_count),
            'cancelled_invoices': int(cancelled_count),
            'invalid_quantity': int(invalid_qty_count),
            'invalid_price': int(invalid_price_count)
        }
    }
    
    with open('logs/data_quality_report.json', 'w') as f:
        json.dump(dq_report, f, indent=4)
    
    print("Validation Data Quality Report:")
    print(json.dumps(dq_report, indent=4))
    return df

if __name__ == '__main__':
    validate_data()

