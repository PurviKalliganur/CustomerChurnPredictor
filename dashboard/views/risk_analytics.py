"""
Customer Risk Analytics
-----------------------
Cohort-level risk segmentation across all 4,338 customer entities.
High-risk VIP table, lifetime spend filters, and targeted CSV export.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils import get_connection

@st.cache_data(ttl=120)
def load_scored_customers():
    conn = get_connection()
    q = "SELECT * FROM customer_churn_scored ORDER BY churn_probability DESC;"
    try:
        return pd.read_sql(q, conn)
    except Exception:
        try:
            return pd.read_csv("data/processed/customer_churn_scored.csv")
        except Exception:
            return pd.DataFrame()

def render():
    df_scored = load_scored_customers()
    if df_scored.empty:
        st.warning("No scored customer records found. Please run batch scoring first.")
        return

    # --------------------------------------------------------------------------
    # COHORT SUMMARY TILES
    # --------------------------------------------------------------------------
    total_custs = len(df_scored)
    high_risk_df = df_scored[df_scored['churn_probability'] >= 0.65]
    mod_risk_df = df_scored[(df_scored['churn_probability'] >= 0.35) & (df_scored['churn_probability'] < 0.65)]
    low_risk_df = df_scored[df_scored['churn_probability'] < 0.35]

    revenue_at_risk = high_risk_df['monetary_total'].sum()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Scored Base</div>
            <div class="metric-panel-value">{total_custs:,}</div>
            <div class="metric-panel-sub">100% of live customer entities</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Critical High Risk</div>
            <div class="metric-panel-value" style="color:#dc2626;">{len(high_risk_df):,}</div>
            <div class="metric-panel-sub">{(len(high_risk_df)/total_custs)*100:.1f}% (&gt;65% probability)</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Moderate Risk</div>
            <div class="metric-panel-value" style="color:#d97706;">{len(mod_risk_df):,}</div>
            <div class="metric-panel-sub">{(len(mod_risk_df)/total_custs)*100:.1f}% (35% - 65% probability)</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Estimated Spend at Risk</div>
            <div class="metric-panel-value" style="color:#dc2626;">${revenue_at_risk:,.2f}</div>
            <div class="metric-panel-sub">Cumulative high-risk spend</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RISK DISTRIBUTION CHARTS
    # --------------------------------------------------------------------------
    ch_col1, ch_col2 = st.columns([1, 1], gap="medium")

    with ch_col1:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Risk Cohort Distribution</div>", unsafe_allow_html=True)
        cohort_summary = pd.DataFrame([
            {"Cohort": "Low Risk (<35%)", "Count": len(low_risk_df), "Revenue": low_risk_df['monetary_total'].sum()},
            {"Cohort": "Moderate Risk (35-65%)", "Count": len(mod_risk_df), "Revenue": mod_risk_df['monetary_total'].sum()},
            {"Cohort": "Critical High Risk (>65%)", "Count": len(high_risk_df), "Revenue": high_risk_df['monetary_total'].sum()}
        ])
        fig_donut = px.pie(
            cohort_summary, names="Cohort", values="Count", hole=0.5,
            color="Cohort",
            color_discrete_map={
                "Low Risk (<35%)": "#10b981",
                "Moderate Risk (35-65%)": "#f59e0b",
                "Critical High Risk (>65%)": "#ef4444"
            }
        )
        fig_donut.update_layout(
            paper_bgcolor='#ffffff', font=dict(color='#475569', family='Inter', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=270
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch_col2:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Cumulative Revenue by Risk Cohort</div>", unsafe_allow_html=True)
        fig_bar = px.bar(
            cohort_summary, x="Cohort", y="Revenue",
            color="Cohort",
            color_discrete_map={
                "Low Risk (<35%)": "#10b981",
                "Moderate Risk (35-65%)": "#f59e0b",
                "Critical High Risk (>65%)": "#ef4444"
            }
        )
        fig_bar.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
            font=dict(color='#475569', family='Inter', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=270,
            yaxis=dict(showgrid=True, gridcolor='#e2e8f0'), xaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # HIGH-RISK VIP TARGET TABLE & EXPORT
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>High-Risk Customer Retention Audience</div>", unsafe_allow_html=True)
    st.caption("Filter and export high-value customers at immediate risk of churn to CRM or marketing retention teams.")

    flt_col1, flt_col2, flt_col3 = st.columns([1, 1, 1], gap="small")
    with flt_col1:
        min_spend = st.selectbox("Minimum Spend Threshold:", [0, 500, 1000, 2500, 5000, 10000], index=1)
    with flt_col2:
        tier_filter = st.selectbox("Risk Filter:", ["Critical High Risk Only", "Moderate + High Risk", "All Customers"])
    with flt_col3:
        search_id = st.text_input("Filter Customer ID:", "", placeholder="Search ID...")

    # Apply filters
    filtered_cohort = df_scored.copy()
    filtered_cohort = filtered_cohort[filtered_cohort['monetary_total'] >= min_spend]

    if tier_filter == "Critical High Risk Only":
        filtered_cohort = filtered_cohort[filtered_cohort['churn_probability'] >= 0.65]
    elif tier_filter == "Moderate + High Risk":
        filtered_cohort = filtered_cohort[filtered_cohort['churn_probability'] >= 0.35]

    if search_id.strip():
        filtered_cohort = filtered_cohort[filtered_cohort['customer_id'].astype(str).str.contains(search_id.strip())]

    st.caption(f"Displaying {len(filtered_cohort):,} matching customer records")
    
    display_cols = [
        "customer_id", "churn_probability", "risk_tier", "monetary_total",
        "recency_days", "frequency", "cancellation_rate", "retention_action"
    ]
    avail_cols = [c for c in display_cols if c in filtered_cohort.columns]

    st.dataframe(
        filtered_cohort[avail_cols],
        column_config={
            "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
            "churn_probability": st.column_config.NumberColumn("Churn Prob", format="%.2%"),
            "risk_tier": "Risk Tier",
            "monetary_total": st.column_config.NumberColumn("Lifetime Spend", format="$%.2f"),
            "recency_days": st.column_config.NumberColumn("Recency", format="%d d"),
            "frequency": st.column_config.NumberColumn("Orders", format="%d"),
            "cancellation_rate": st.column_config.NumberColumn("Cancel Rate", format="%.1%"),
            "retention_action": "Retention Playbook Action"
        },
        hide_index=True,
        use_container_width=True
    )

    # Export Button
    csv_audience = filtered_cohort.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Export Filtered Audience ({len(filtered_cohort):,} Customers) to CSV",
        data=csv_audience,
        file_name="churn_retention_target_audience.csv",
        mime="text/csv",
        type="primary"
    )
