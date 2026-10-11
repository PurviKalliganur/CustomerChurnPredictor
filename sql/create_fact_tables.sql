-- create_fact_tables.sql
-- Fact Table: sales

CREATE TABLE IF NOT EXISTS fact_sales (
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(50) NOT NULL,
    customer_id INT,
    invoice_date TIMESTAMP NOT NULL,
    
    date_sk INT REFERENCES dim_date(date_sk),
    product_sk INT REFERENCES dim_product(product_sk),
    customer_sk INT REFERENCES dim_customer(customer_sk),
    geo_sk INT REFERENCES dim_geography(geo_sk),
    
    quantity INT NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    revenue NUMERIC(10, 2) NOT NULL,
    
    -- Business Key to prevent duplicates
    CONSTRAINT pk_fact_sales PRIMARY KEY (invoice_no, stock_code, invoice_date)
);
