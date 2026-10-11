"""
Business Intelligence & Analytics
----------------------------------
E-commerce analytics workspace synthesized from fact_sales and customer_mart.
Visualizes revenue trends, geographic distributions, product volume, and RFM distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.utils import get_connection

@st.cache_data(ttl=120)
def load_bi_data():
    conn = get_connection()
    q = """
    SELECT 
        f.invoice_no,
        d.full_date,
        d.year,
        d.month,
        g.country,
        p.description AS product_description,
        f.quantity,
        f.revenue,
        f.customer_id
    FROM fact_sales f
    JOIN dim_date d ON f.date_sk = d.date_sk
    JOIN dim_geography g ON f.geo_sk = g.geo_sk
    JOIN dim_product p ON f.product_sk = p.product_sk;
    """
    df = pd.read_sql(q, conn)
    df['full_date'] = pd.to_datetime(df['full_date'])
    return df

@st.cache_data(ttl=120)
def load_rfm_data():
    conn = get_connection()
    q = "SELECT customer_id, country, total_revenue, total_orders, rfm_recency, rfm_frequency, rfm_monetary, repeat_purchase_flag, avg_order_value FROM customer_mart;"
    try:
        df = pd.read_sql(q, conn)
        def assign_seg(r):
            rec = r['rfm_recency']
            freq = r['rfm_frequency']
            mon = r['rfm_monetary']
            if rec <= 30 and freq >= 4:
                return 'Champions / VIP'
            elif rec <= 60 and freq >= 2:
                return 'Potential Loyalists'
            elif rec > 120 and mon >= 1000:
                return 'At Risk (High Spend)'
            elif rec > 180:
                return 'Hibernating / Churned'
            else:
                return 'Regular Active'
        df['customer_segment'] = df.apply(assign_seg, axis=1)
        return df
    except Exception:
        return pd.DataFrame()

def render():
    with st.spinner("Loading analytical facts from warehouse..."):
        sales_df = load_bi_data()
        rfm_df = load_rfm_data()

    if sales_df.empty:
        st.warning("No sales data found in warehouse.")
        return

    # --------------------------------------------------------------------------
    # GLOBAL FILTER TOOLBAR
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Analytics Filter Toolbar</div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2], gap="small")
    
    min_date = sales_df['full_date'].min()
    max_date = sales_df['full_date'].max()

    with f_col1:
        date_sel = st.date_input("Date Window:", [min_date, max_date], min_value=min_date, max_value=max_date, label_visibility="collapsed")
    with f_col2:
        all_countries = sorted(sales_df['country'].dropna().unique().tolist())
        selected_countries = st.multiselect("Filter Countries:", options=all_countries, default=[], placeholder="All Countries (Global)", label_visibility="collapsed")
    with f_col3:
        top_prods = sales_df.groupby('product_description')['revenue'].sum().nlargest(25).index.tolist()
        selected_prods = st.multiselect("Filter Top Products:", options=top_prods, default=[], placeholder="All Catalog Products", label_visibility="collapsed")

    # Apply filters
    filtered_df = sales_df.copy()
    if len(date_sel) == 2:
        start_d, end_d = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
        filtered_df = filtered_df[(filtered_df['full_date'] >= start_d) & (filtered_df['full_date'] <= end_d)]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_prods:
        filtered_df = filtered_df[filtered_df['product_description'].isin(selected_prods)]

    # --------------------------------------------------------------------------
    # KPI TILES
    # --------------------------------------------------------------------------
    rev = filtered_df['revenue'].sum()
    orders = filtered_df['invoice_no'].nunique()
    custs = filtered_df['customer_id'].nunique()
    aov = (rev / orders) if orders > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Filtered Revenue</div>
            <div class="metric-panel-value">${rev:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Completed Orders</div>
            <div class="metric-panel-value">{orders:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Active Customers</div>
            <div class="metric-panel-value">{custs:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Average Order Value</div>
            <div class="metric-panel-value">${aov:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # VISUALIZATION ROW 1: MONTHLY TRAJECTORY & GEOGRAPHY
    # --------------------------------------------------------------------------
    v_col1, v_col2 = st.columns(2, gap="medium")

    with v_col1:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Monthly Revenue Trajectory</div>", unsafe_allow_html=True)
        monthly_df = filtered_df.groupby([filtered_df['full_date'].dt.to_period('M')]).agg(
            revenue=('revenue', 'sum'),
            orders=('invoice_no', 'nunique')
        ).reset_index()
        monthly_df['period'] = monthly_df['full_date'].astype(str)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_df['period'], y=monthly_df['revenue'],
            name='Revenue ($)', fill='tozeroy', line=dict(color='#2563eb', width=2)
        ))
        fig_trend.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
            font=dict(color='#475569', family='Inter', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with v_col2:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Geographic Revenue Distribution (Top 10)</div>", unsafe_allow_html=True)
        geo_df = filtered_df.groupby('country')['revenue'].sum().nlargest(10).reset_index()
        fig_geo = px.bar(
            geo_df, x='revenue', y='country', orientation='h',
            color='revenue', color_continuous_scale='Blues'
        )
        fig_geo.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
            font=dict(color='#475569', family='Inter', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    # --------------------------------------------------------------------------
    # VISUALIZATION ROW 2: TOP PRODUCTS & RFM SEGMENTATION
    # --------------------------------------------------------------------------
    v_col3, v_col4 = st.columns(2, gap="medium")

    with v_col3:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Top 10 Revenue-Generating Catalog Products</div>", unsafe_allow_html=True)
        prod_df = filtered_df.groupby('product_description')['revenue'].sum().nlargest(10).reset_index()
        fig_prod = px.bar(
            prod_df, x='revenue', y='product_description', orientation='h',
            color='revenue', color_continuous_scale='Teal'
        )
        fig_prod.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
            font=dict(color='#475569', family='Inter', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=310,
            yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with v_col4:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>RFM Customer Cohort Segmentation</div>", unsafe_allow_html=True)
        if not rfm_df.empty and 'customer_segment' in rfm_df.columns:
            seg_df = rfm_df['customer_segment'].value_counts().reset_index()
            seg_df.columns = ['segment', 'count']
            fig_rfm = px.pie(
                seg_df, names='segment', values='count', hole=0.45,
                color_discrete_sequence=['#2563eb', '#38bdf8', '#10b981', '#f59e0b', '#64748b']
            )
            fig_rfm.update_layout(
                paper_bgcolor='#ffffff', font=dict(color='#475569', family='Inter', size=11),
                margin=dict(l=10, r=10, t=10, b=10), height=310
            )
            st.plotly_chart(fig_rfm, use_container_width=True)
        else:
            st.info("RFM segmentation mart not populated yet.")
