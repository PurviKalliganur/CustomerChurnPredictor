-- create_dimensions.sql
-- Dimensions: customer, product, date, geography

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_sk SERIAL PRIMARY KEY,
    customer_id INT UNIQUE NOT NULL,
    rfm_recency INT,
    rfm_frequency INT,
    rfm_monetary NUMERIC(10, 2),
    repeat_purchase_flag BOOLEAN,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_sk SERIAL PRIMARY KEY,
    stock_code VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(255),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_sk INT PRIMARY KEY, -- Format YYYYMMDD
    full_date DATE UNIQUE NOT NULL,
    day_of_week INT,
    day_name VARCHAR(15),
    day_of_month INT,
    month INT,
    month_name VARCHAR(15),
    quarter INT,
    year INT
);

CREATE TABLE IF NOT EXISTS dim_geography (
    geo_sk SERIAL PRIMARY KEY,
    country VARCHAR(100) UNIQUE NOT NULL
);
