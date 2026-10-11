"""
Data Warehouse Explorer
-----------------------
Kimball-style dimensional warehouse inspection workspace.
Browse fact and dimension tables, inspect column schemas, and execute read-only analytical SQL queries.
"""

import streamlit as st
import pandas as pd
from dashboard.utils import (
    get_all_tables, get_table_schema, load_table_data, execute_safe_sql, get_db_info
)

EXAMPLE_QUERIES = {
    "Top 10 High-Spend Customers": """
SELECT 
    c.customer_id,
    c.country,
    COUNT(DISTINCT f.invoice_no) AS total_orders,
    ROUND(SUM(f.revenue), 2) AS lifetime_spend,
    ROUND(AVG(f.revenue), 2) AS avg_item_revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
GROUP BY c.customer_id, c.country
ORDER BY lifetime_spend DESC
LIMIT 10;
""",
    "Monthly Revenue & Order Trend": """
SELECT 
    d.year,
    d.month,
    COUNT(DISTINCT f.invoice_no) AS order_volume,
    ROUND(SUM(f.revenue), 2) AS monthly_revenue,
    ROUND(SUM(f.revenue) / COUNT(DISTINCT f.invoice_no), 2) AS monthly_aov
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
GROUP BY d.year, d.month
ORDER BY d.year, d.month;
""",
    "High Risk Customer Cohort Preview": """
SELECT 
    customer_id,
    churn_probability,
    risk_tier,
    monetary_total,
    recency_days,
    frequency,
    retention_action
FROM customer_churn_scored
WHERE churn_probability >= 0.65
ORDER BY monetary_total DESC
LIMIT 15;
""",
    "Sales Distribution by Country": """
SELECT 
    g.country,
    COUNT(DISTINCT f.customer_id) AS customer_count,
    COUNT(DISTINCT f.invoice_no) AS total_orders,
    ROUND(SUM(f.revenue), 2) AS total_revenue
FROM fact_sales f
JOIN dim_geography g ON f.geo_sk = g.geo_sk
GROUP BY g.country
ORDER BY total_revenue DESC
LIMIT 10;
"""
}

def render():
    db_type, db_size, _ = get_db_info()
    df_tables = get_all_tables()

    # --------------------------------------------------------------------------
    # WAREHOUSE CATALOG SUMMARY
    # --------------------------------------------------------------------------
    cat_col1, cat_col2 = st.columns([1, 2], gap="medium")
    with cat_col1:
        st.markdown(f"""
        <div class="metric-panel" style="margin-bottom:12px;">
            <div class="metric-panel-label">Warehouse Storage Engine</div>
            <div class="metric-panel-value" style="font-size:18px;">{db_type}</div>
            <div class="metric-panel-sub">Active database size: {db_size}</div>
        </div>
        """, unsafe_allow_html=True)
    with cat_col2:
        st.markdown("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Warehouse Tables & Views Catalog</div>", unsafe_allow_html=True)
        st.dataframe(df_tables, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TABLE & SCHEMA INSPECTOR
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Table & Schema Inspector</div>", unsafe_allow_html=True)

    t_col1, t_col2, t_col3 = st.columns([2, 1.5, 1], gap="small")
    with t_col1:
        table_list = df_tables['name'].tolist() if not df_tables.empty else ["fact_sales"]
        selected_table = st.selectbox("Select Table or View:", table_list, index=table_list.index("fact_sales") if "fact_sales" in table_list else 0, label_visibility="collapsed")
    with t_col2:
        search_kw = st.text_input("Filter Rows:", "", placeholder="Search text columns...", label_visibility="collapsed")
    with t_col3:
        page_limit = st.selectbox("Row Limit:", [25, 50, 100, 200], index=1, label_visibility="collapsed")

    with st.expander(f"View Schema Definition for `{selected_table}`", expanded=False):
        df_schema = get_table_schema(selected_table)
        st.dataframe(df_schema, use_container_width=True, hide_index=True)

    df_sample = load_table_data(selected_table, limit=page_limit, search_query=search_kw)
    st.caption(f"Previewing top {len(df_sample)} records from `{selected_table}`")
    st.dataframe(df_sample, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SAFE READ-ONLY SQL QUERY STUDIO
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Read-Only Analytical SQL Studio</div>", unsafe_allow_html=True)
    st.caption("Execute custom analytical queries under read-only safety guardrails (DDL/DML operations blocked).")

    ex_choice = st.selectbox("Load Prebuilt SQL Template:", ["-- Select a template --"] + list(EXAMPLE_QUERIES.keys()))
    default_sql = EXAMPLE_QUERIES.get(ex_choice, "SELECT * FROM fact_sales LIMIT 25;")

    user_query = st.text_area("SQL Editor:", value=default_sql, height=130, label_visibility="collapsed")

    q_col1, q_col2 = st.columns([1.5, 4.5], gap="small")
    with q_col1:
        run_query_btn = st.button("Execute Query", type="primary", use_container_width=True)

    if run_query_btn:
        df_res, duration_ms, error = execute_safe_sql(user_query)
        if error:
            st.error(f"SQL Error: {error}")
        else:
            st.success(f"Query returned {len(df_res):,} rows in {duration_ms:.2f} ms")
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            csv_res = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Query Result to CSV",
                data=csv_res,
                file_name="sql_query_result.csv",
                mime="text/csv"
            )
