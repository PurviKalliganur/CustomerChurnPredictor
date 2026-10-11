-- create_schema.sql
-- Initializes the basic metadata table and any necessary extensions.

CREATE TABLE IF NOT EXISTS etl_metadata (
    run_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_processed_date TIMESTAMP,
    row_count INT,
    status VARCHAR(50)
);
