import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure working directory is project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/load.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Optional psycopg2 support
try:
    import psycopg2
    from psycopg2.extras import execute_values
    from psycopg2.extensions import register_adapter, AsIs
    register_adapter(np.int64, AsIs)
    register_adapter(np.float64, AsIs)
    register_adapter(np.bool_, AsIs)
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

def get_postgres_connection():
    if not POSTGRES_AVAILABLE:
        raise ImportError("psycopg2 is not installed.")
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', 5432),
        database=os.getenv('POSTGRES_DB', 'ecommerce'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
        connect_timeout=3
    )
    return conn

def clean_df(df):
    for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
        df[col] = df[col].dt.to_pydatetime()
    return df.astype(object).where(pd.notnull(df), None)

def load_to_postgres():
    logging.info("Starting data load to PostgreSQL...")
    conn = get_postgres_connection()
    cur = conn.cursor()
    try:
        # 0. Auto-initialize schema from SQL files if not already done
        for sql_file in ['sql/create_schema.sql', 'sql/create_dimensions.sql', 'sql/create_fact_tables.sql', 'sql/create_marts.sql']:
            if os.path.exists(sql_file):
                with open(sql_file, 'r') as f:
                    cur.execute(f.read())
        conn.commit()

        # 1. Load dim_geography
        dim_geo = pd.read_csv('data/processed/dim_geography.csv')
        dim_geo = clean_df(dim_geo)
        geo_records = dim_geo.to_records(index=False)
        execute_values(cur, """
            INSERT INTO dim_geography (country) VALUES %s
            ON CONFLICT (country) DO NOTHING;
        """, geo_records)
        
        # 2. Load dim_product
        dim_prod = pd.read_csv('data/processed/dim_product.csv')
        dim_prod['Description'] = dim_prod['Description'].fillna('')
        dim_prod = clean_df(dim_prod[['StockCode', 'Description']])
        prod_records = dim_prod.to_records(index=False)
        execute_values(cur, """
            INSERT INTO dim_product (stock_code, description) VALUES %s
            ON CONFLICT (stock_code) DO UPDATE 
            SET description = EXCLUDED.description, last_updated = CURRENT_TIMESTAMP;
        """, prod_records)
        
        # 3. Load dim_customer
        dim_cust = pd.read_csv('data/processed/dim_customer.csv')
        dim_cust = clean_df(dim_cust[['CustomerID', 'rfm_recency', 'rfm_frequency', 'rfm_monetary', 'repeat_purchase_flag']])
        cust_records = dim_cust.to_records(index=False)
        execute_values(cur, """
            INSERT INTO dim_customer (customer_id, rfm_recency, rfm_frequency, rfm_monetary, repeat_purchase_flag) VALUES %s
            ON CONFLICT (customer_id) DO UPDATE 
            SET rfm_recency = EXCLUDED.rfm_recency, 
                rfm_frequency = EXCLUDED.rfm_frequency,
                rfm_monetary = EXCLUDED.rfm_monetary,
                repeat_purchase_flag = EXCLUDED.repeat_purchase_flag,
                last_updated = CURRENT_TIMESTAMP;
        """, cust_records)
        
        # 4. Load dim_date
        fact_sales = pd.read_csv('data/processed/fact_sales.csv')
        fact_sales['InvoiceDate'] = pd.to_datetime(fact_sales['InvoiceDate'])
        unique_dates = fact_sales['InvoiceDate'].dt.date.unique()
        date_records = []
        for d in unique_dates:
            date_sk = int(d.strftime('%Y%m%d'))
            day_of_week = d.weekday()
            day_name = d.strftime('%A')
            day_of_month = d.day
            month = d.month
            month_name = d.strftime('%B')
            quarter = (d.month - 1) // 3 + 1
            year = d.year
            date_records.append((date_sk, d, day_of_week, day_name, day_of_month, month, month_name, quarter, year))
            
        execute_values(cur, """
            INSERT INTO dim_date (date_sk, full_date, day_of_week, day_name, day_of_month, month, month_name, quarter, year) 
            VALUES %s
            ON CONFLICT (full_date) DO NOTHING;
        """, date_records)
        
        # 5. Load fact_sales
        fact_sales['date_sk'] = fact_sales['InvoiceDate'].dt.strftime('%Y%m%d').astype(int)
        cur.execute("CREATE TEMP TABLE stg_fact_sales (invoice_no VARCHAR, stock_code VARCHAR, customer_id INT, invoice_date TIMESTAMP, country VARCHAR, quantity INT, unit_price NUMERIC, revenue NUMERIC, date_sk INT)")
        fact_sales_subset = clean_df(fact_sales[['InvoiceNo', 'StockCode', 'CustomerID', 'InvoiceDate', 'Country', 'Quantity', 'UnitPrice', 'Revenue', 'date_sk']])
        fact_records = fact_sales_subset.to_records(index=False)
        execute_values(cur, """
            INSERT INTO stg_fact_sales (invoice_no, stock_code, customer_id, invoice_date, country, quantity, unit_price, revenue, date_sk) VALUES %s
        """, fact_records)
        
        cur.execute("""
            INSERT INTO fact_sales (invoice_no, stock_code, customer_id, invoice_date, date_sk, product_sk, customer_sk, geo_sk, quantity, unit_price, revenue)
            SELECT 
                s.invoice_no, s.stock_code, s.customer_id, s.invoice_date, s.date_sk,
                p.product_sk, c.customer_sk, g.geo_sk, s.quantity, s.unit_price, s.revenue
            FROM stg_fact_sales s
            LEFT JOIN dim_product p ON s.stock_code = p.stock_code
            LEFT JOIN dim_customer c ON s.customer_id = c.customer_id
            LEFT JOIN dim_geography g ON s.country = g.country
            ON CONFLICT (invoice_no, stock_code, invoice_date) DO UPDATE
            SET quantity = EXCLUDED.quantity, unit_price = EXCLUDED.unit_price, revenue = EXCLUDED.revenue;
        """)
        
        # 6. Metadata update
        cur.execute("""
            INSERT INTO etl_metadata (pipeline_name, last_processed_date, row_count, status)
            VALUES (%s, %s, %s, %s)
        """, ('ecommerce_etl_pipeline', datetime.now(), len(fact_sales), 'success'))
        
        conn.commit()
        logging.info("Load to PostgreSQL successful.")
        print("Data loaded to PostgreSQL successfully.")
    finally:
        cur.close()
        conn.close()

def init_sqlite_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS etl_metadata (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_name TEXT NOT NULL,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_processed_date TIMESTAMP,
            row_count INTEGER,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS dim_customer (
            customer_sk INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER UNIQUE NOT NULL,
            rfm_recency INTEGER,
            rfm_frequency INTEGER,
            rfm_monetary REAL,
            repeat_purchase_flag BOOLEAN,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dim_product (
            product_sk INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT UNIQUE NOT NULL,
            description TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dim_date (
            date_sk INTEGER PRIMARY KEY,
            full_date TEXT UNIQUE NOT NULL,
            day_of_week INTEGER,
            day_name TEXT,
            day_of_month INTEGER,
            month INTEGER,
            month_name TEXT,
            quarter INTEGER,
            year INTEGER
        );

        CREATE TABLE IF NOT EXISTS dim_geography (
            geo_sk INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fact_sales (
            invoice_no TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            customer_id INTEGER,
            invoice_date TIMESTAMP NOT NULL,
            date_sk INTEGER REFERENCES dim_date(date_sk),
            product_sk INTEGER REFERENCES dim_product(product_sk),
            customer_sk INTEGER REFERENCES dim_customer(customer_sk),
            geo_sk INTEGER REFERENCES dim_geography(geo_sk),
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            revenue REAL NOT NULL,
            PRIMARY KEY (invoice_no, stock_code, invoice_date)
        );

        CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date_sk);
        CREATE INDEX IF NOT EXISTS idx_fact_sales_geo ON fact_sales(geo_sk);
        CREATE INDEX IF NOT EXISTS idx_fact_sales_product ON fact_sales(product_sk);
        CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON fact_sales(customer_sk);

        DROP VIEW IF EXISTS sales_mart;
        CREATE VIEW sales_mart AS
        SELECT 
            d.full_date,
            g.country,
            p.description AS product_description,
            SUM(f.quantity) AS total_quantity,
            SUM(f.revenue) AS total_revenue,
            COUNT(DISTINCT f.invoice_no) AS total_orders,
            COUNT(DISTINCT f.customer_id) AS unique_customers
        FROM fact_sales f
        JOIN dim_date d ON f.date_sk = d.date_sk
        JOIN dim_geography g ON f.geo_sk = g.geo_sk
        JOIN dim_product p ON f.product_sk = p.product_sk
        GROUP BY d.full_date, g.country, p.description;

        DROP VIEW IF EXISTS customer_mart;
        CREATE VIEW customer_mart AS
        SELECT 
            c.customer_id,
            g.country,
            SUM(f.revenue) AS total_revenue,
            COUNT(DISTINCT f.invoice_no) AS total_orders,
            c.rfm_recency,
            c.rfm_frequency,
            c.rfm_monetary,
            c.repeat_purchase_flag,
            CASE 
                WHEN COUNT(DISTINCT f.invoice_no) > 0 THEN SUM(f.revenue) / COUNT(DISTINCT f.invoice_no)
                ELSE 0 
            END AS avg_order_value
        FROM dim_customer c
        LEFT JOIN fact_sales f ON c.customer_sk = f.customer_sk
        LEFT JOIN dim_geography g ON f.geo_sk = g.geo_sk
        GROUP BY 
            c.customer_id, 
            g.country, 
            c.rfm_recency, 
            c.rfm_frequency, 
            c.rfm_monetary, 
            c.repeat_purchase_flag;
    """)
    conn.commit()

def load_to_sqlite(db_path='ecommerce.db'):
    logging.info(f"Loading data into SQLite warehouse ({db_path})...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Enable performance settings for fast bulk loading
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
    
    init_sqlite_schema(conn)
    
    try:
        # 1. dim_geography
        dim_geo = pd.read_csv('data/processed/dim_geography.csv')
        cur.executemany("INSERT OR IGNORE INTO dim_geography (country) VALUES (?);", [(r,) for r in dim_geo['Country'].dropna().unique()])
        conn.commit()
        
        # 2. dim_product
        dim_prod = pd.read_csv('data/processed/dim_product.csv')
        dim_prod['Description'] = dim_prod['Description'].fillna('')
        prod_records = dim_prod[['StockCode', 'Description']].drop_duplicates(subset=['StockCode']).to_records(index=False).tolist()
        cur.executemany("INSERT OR REPLACE INTO dim_product (stock_code, description) VALUES (?, ?);", prod_records)
        conn.commit()
        
        # 3. dim_customer
        dim_cust = pd.read_csv('data/processed/dim_customer.csv')
        cust_records = dim_cust[['CustomerID', 'rfm_recency', 'rfm_frequency', 'rfm_monetary', 'repeat_purchase_flag']].to_records(index=False).tolist()
        cur.executemany("""
            INSERT OR REPLACE INTO dim_customer (customer_id, rfm_recency, rfm_frequency, rfm_monetary, repeat_purchase_flag)
            VALUES (?, ?, ?, ?, ?);
        """, cust_records)
        conn.commit()
        
        # 4. dim_date
        fact_sales = pd.read_csv('data/processed/fact_sales.csv')
        fact_sales['InvoiceDate'] = pd.to_datetime(fact_sales['InvoiceDate'])
        unique_dates = fact_sales['InvoiceDate'].dt.date.unique()
        date_records = []
        for d in unique_dates:
            date_sk = int(d.strftime('%Y%m%d'))
            day_of_week = int(d.weekday())
            day_name = d.strftime('%A')
            day_of_month = int(d.day)
            month = int(d.month)
            month_name = d.strftime('%B')
            quarter = int((d.month - 1) // 3 + 1)
            year = int(d.year)
            date_records.append((date_sk, str(d), day_of_week, day_name, day_of_month, month, month_name, quarter, year))
            
        cur.executemany("""
            INSERT OR IGNORE INTO dim_date (date_sk, full_date, day_of_week, day_name, day_of_month, month, month_name, quarter, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, date_records)
        conn.commit()
        
        # 5. fact_sales with surrogate key mapping
        geo_map = dict(cur.execute("SELECT country, geo_sk FROM dim_geography").fetchall())
        prod_map = dict(cur.execute("SELECT stock_code, product_sk FROM dim_product").fetchall())
        cust_map = dict(cur.execute("SELECT customer_id, customer_sk FROM dim_customer").fetchall())
        
        fact_sales['geo_sk'] = fact_sales['Country'].map(geo_map)
        fact_sales['product_sk'] = fact_sales['StockCode'].map(prod_map)
        fact_sales['customer_sk'] = fact_sales['CustomerID'].map(cust_map)
        fact_sales['date_sk'] = fact_sales['InvoiceDate'].dt.strftime('%Y%m%d').astype(int)
        
        cols = ['InvoiceNo', 'StockCode', 'CustomerID', 'InvoiceDate', 'date_sk', 'product_sk', 'customer_sk', 'geo_sk', 'Quantity', 'UnitPrice', 'Revenue']
        fact_subset = fact_sales[cols].copy()
        fact_subset['InvoiceDate'] = fact_subset['InvoiceDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
        fact_records = fact_subset.to_records(index=False).tolist()
        
        cur.executemany("""
            INSERT OR REPLACE INTO fact_sales 
            (invoice_no, stock_code, customer_id, invoice_date, date_sk, product_sk, customer_sk, geo_sk, quantity, unit_price, revenue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, fact_records)
        
        # 6. etl_metadata
        cur.execute("""
            INSERT INTO etl_metadata (pipeline_name, last_processed_date, row_count, status)
            VALUES (?, ?, ?, ?)
        """, ('ecommerce_etl_pipeline', datetime.now().isoformat(), len(fact_sales), 'success'))
        
        conn.commit()
        logging.info(f"Loaded {len(fact_sales)} fact rows into SQLite successfully.")
        print(f"Data loaded into SQLite ({db_path}) successfully. Total fact rows: {len(fact_sales)}")
    finally:
        conn.close()

def load_data():
    load_dotenv()
    use_sqlite = os.getenv('USE_SQLITE', 'false').lower() in ('true', '1', 't')
    
    if not use_sqlite and POSTGRES_AVAILABLE:
        try:
            load_to_postgres()
            return
        except Exception as e:
            print(f"PostgreSQL connection unavailable ({e}).")
            print("Falling back to local SQLite warehouse (ecommerce.db)...")
            logging.warning(f"PostgreSQL load failed, falling back to SQLite: {e}")
    
    # Fallback to SQLite
    load_to_sqlite()

if __name__ == '__main__':
    load_data()
