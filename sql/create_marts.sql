-- create_marts.sql
-- Creates views for analytical marts to be consumed by Streamlit

-- 1. Sales Mart
-- Contains aggregated daily sales data
CREATE OR REPLACE VIEW sales_mart AS
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

-- 2. Customer Mart
-- Contains customer-level aggregated metrics
CREATE OR REPLACE VIEW customer_mart AS
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
