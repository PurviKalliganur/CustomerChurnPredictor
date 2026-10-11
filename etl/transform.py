import os
import sys
import logging
from datetime import timedelta
import pandas as pd

# Ensure working directory is project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

logging.basicConfig(
    filename='logs/transform.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def transform_data():
    input_path = 'data/processed/val_online_retail.csv'
    fact_path = 'data/processed/fact_sales.csv'
    customer_path = 'data/processed/dim_customer.csv'
    product_path = 'data/processed/dim_product.csv'
    geo_path = 'data/processed/dim_geography.csv'
    
    logging.info("Starting data transformation...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Validated dataset not found at {input_path}. Please run validation step first.")
        
    df = pd.read_csv(input_path)
    
    # 1. Standardize InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 2. Calculate Revenue
    df['Revenue'] = (df['Quantity'] * df['UnitPrice']).round(2)
    
    # 3. Create Fact Sales (Granularity: line item)
    # Ensure CustomerID is int
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    fact_sales = df.groupby(['InvoiceNo', 'StockCode', 'InvoiceDate', 'CustomerID', 'Country'], as_index=False).agg({
        'Quantity': 'sum',
        'UnitPrice': lambda x: round(x.mean(), 2),
        'Revenue': lambda x: round(x.sum(), 2)
    })
    
    # 4. Dimension: Product
    # Extract unique products and sanitize null descriptions
    dim_product = df[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'], keep='last').copy()
    dim_product['Description'] = dim_product['Description'].fillna('')
    
    # 5. Customer RFM Calculation
    # Reference Date: max date in dataset + 1 day
    ref_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                             # Frequency
        'Revenue': lambda x: round(x.sum(), 2)              # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'rfm_recency',
        'InvoiceNo': 'rfm_frequency',
        'Revenue': 'rfm_monetary'
    }, inplace=True)
    
    # Repeat purchase flag
    rfm['repeat_purchase_flag'] = rfm['rfm_frequency'] > 1
    dim_customer = rfm
    
    # 6. Geography Dimension
    dim_geography = df[['Country']].drop_duplicates()
    
    # Save transformed data for loading
    fact_sales.to_csv(fact_path, index=False)
    dim_customer.to_csv(customer_path, index=False)
    dim_product.to_csv(product_path, index=False)
    dim_geography.to_csv(geo_path, index=False)
    
    logging.info(f"Transformation complete. Fact sales: {len(fact_sales)}, Customers: {len(dim_customer)}, Products: {len(dim_product)}")
    print(f"Transformation successful. Fact rows: {len(fact_sales)}, Customers: {len(dim_customer)}, Products: {len(dim_product)}")
    return fact_sales, dim_customer, dim_product, dim_geography

if __name__ == '__main__':
    transform_data()

