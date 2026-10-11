import os
import sys
import time
import logging

# Ensure project root is the working directory and etl is in sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'etl'))
os.makedirs('logs', exist_ok=True)

from ingest import ingest_data
from validate import validate_data
from transform import transform_data
from load import load_data

logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_pipeline():
    start_total = time.time()
    print("=========================================================")
    print("       E-Commerce Sales Analytics ETL Pipeline           ")
    print("=========================================================")
    logging.info("Pipeline started.")
    
    try:
        # Step 1: Ingestion
        print("\n[Step 1/4] Ingestion...")
        t0 = time.time()
        df_raw = ingest_data()
        print(f"--> Ingestion completed in {time.time() - t0:.2f}s (Rows: {len(df_raw):,})")
        
        # Step 2: Validation
        print("\n[Step 2/4] Validation & Data Quality Checks...")
        t0 = time.time()
        df_valid = validate_data()
        print(f"--> Validation completed in {time.time() - t0:.2f}s (Valid: {len(df_valid):,})")
        
        # Step 3: Transformation
        print("\n[Step 3/4] Transformation & Dimensional Modeling...")
        t0 = time.time()
        fact_sales, dim_cust, dim_prod, dim_geo = transform_data()
        print(f"--> Transformation completed in {time.time() - t0:.2f}s (Facts: {len(fact_sales):,}, Customers: {len(dim_cust):,}, Products: {len(dim_prod):,})")
        
        # Step 4: Load to Data Warehouse
        print("\n[Step 4/4] Loading to Data Warehouse...")
        t0 = time.time()
        load_data()
        print(f"--> Data Warehouse load completed in {time.time() - t0:.2f}s")
        
        total_time = time.time() - start_total
        print("\n=========================================================")
        print(f" SUCCESS: Pipeline completed in {total_time:.2f} seconds!")
        print(f" - Raw Input Rows      : {len(df_raw):,}")
        print(f" - Clean Processed Rows: {len(df_valid):,}")
        print(f" - Fact Sales Rows     : {len(fact_sales):,}")
        print(f" - Dimension Customers : {len(dim_cust):,}")
        print(f" - Dimension Products  : {len(dim_prod):,}")
        print("=========================================================")
        logging.info(f"Pipeline execution completed successfully in {total_time:.2f}s.")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    run_pipeline()
