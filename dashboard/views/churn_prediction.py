"""
Customer Churn Prediction
-------------------------
Interactive inference console featuring:
- Mode A: Existing Customer Warehouse Lookup (queries SQLite customer_features)
- Mode B: What-If Behavioral Feature Simulator (13 calibrated sliders & real-time scoring)
- Prescriptive retention playbooks and REST API payload inspector.
"""

import json
import streamlit as st
import pandas as pd
from dashboard.utils import get_connection, predict_customer

@st.cache_data(ttl=60)
def get_sample_customers():
    conn = get_connection()
    q = "SELECT customer_id FROM customer_features ORDER BY customer_id ASC;"
    try:
        df = pd.read_sql(q, conn)
        return df['customer_id'].tolist()
    except Exception:
        return [12346, 12347, 12348, 17850, 13047]

def fetch_customer_features(customer_id):
    conn = get_connection()
    q = f"SELECT * FROM customer_features WHERE customer_id = {int(customer_id)};"
    try:
        df = pd.read_sql(q, conn)
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None

def render():
    st.markdown("<div style='font-size:12px; font-weight:600; text-transform:uppercase; color:#64748b; margin-bottom:4px; letter-spacing:0.5px;'>Inference Workflow</div>", unsafe_allow_html=True)
    
    mode = st.radio(
        "Prediction Mode",
        ["Mode A: Existing Customer Warehouse Lookup", "Mode B: What-If Behavioral Feature Simulator"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # MODE A: EXISTING CUSTOMER LOOKUP
    # ==========================================================================
    if mode.startswith("Mode A"):
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Mode A: Customer Warehouse Lookup</div>", unsafe_allow_html=True)
        st.caption("Retrieve actual behavioral and transactional features directly from the customer feature store.")

        cust_list = get_sample_customers()
        c_sel_col1, c_sel_col2 = st.columns([2, 1], gap="small")
        with c_sel_col1:
            selected_cust = st.selectbox(
                "Select Customer ID:",
                options=cust_list[:500],
                index=0,
                label_visibility="collapsed"
            )
        with c_sel_col2:
            custom_id_in = st.text_input("Or enter exact Customer ID:", "", placeholder="e.g. 17850", label_visibility="collapsed")
            if custom_id_in.strip().isdigit():
                selected_cust = int(custom_id_in.strip())

        cust_feats = fetch_customer_features(selected_cust)

        if cust_feats:
            st.markdown(f"<div style='font-size:12px; font-weight:600; color:#0f172a; margin-top:10px; margin-bottom:6px;'>Feature Snapshot for Customer #{selected_cust}</div>", unsafe_allow_html=True)
            
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                st.markdown(f"<div class='metric-panel'><div class='metric-panel-label'>Recency</div><div class='metric-panel-value' style='font-size:18px;'>{cust_feats.get('recency_days', 0):.0f} d</div></div>", unsafe_allow_html=True)
            with f2:
                st.markdown(f"<div class='metric-panel'><div class='metric-panel-label'>Order Count</div><div class='metric-panel-value' style='font-size:18px;'>{cust_feats.get('frequency', 0):.0f}</div></div>", unsafe_allow_html=True)
            with f3:
                st.markdown(f"<div class='metric-panel'><div class='metric-panel-label'>Spend</div><div class='metric-panel-value' style='font-size:18px;'>${cust_feats.get('monetary_total', 0):,.2f}</div></div>", unsafe_allow_html=True)
            with f4:
                st.markdown(f"<div class='metric-panel'><div class='metric-panel-label'>Basket Size</div><div class='metric-panel-value' style='font-size:18px;'>{cust_feats.get('avg_basket_size', 0):.1f}</div></div>", unsafe_allow_html=True)
            with f5:
                st.markdown(f"<div class='metric-panel'><div class='metric-panel-label'>Cancel Rate</div><div class='metric-panel-value' style='font-size:18px;'>{cust_feats.get('cancellation_rate', 0)*100:.1f}%</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            score_btn = st.button("Score Customer Churn Probability", type="primary")

            if score_btn:
                res = predict_customer(cust_feats)
                prob = res["churn_probability"]
                tier = res["risk_tier"]
                badge = res["badge_class"]
                action = res["retention_playbook"]

                st.markdown(f"""
                <div class="panel-card" style="margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:14px; font-weight:700; color:#0f172a;">Inference Outcome — Customer #{selected_cust}</span>
                        <span class="badge {badge}">{tier}</span>
                    </div>
                    <div style="margin-top:8px; display:flex; align-items:baseline; gap:10px;">
                        <span style="font-size:26px; font-weight:700; color:#2563eb;">{prob*100:.2f}%</span>
                        <span style="font-size:12px; color:#64748b;">Predicted Probability of Churn within next 90 days</span>
                    </div>
                    <div style="margin-top:10px; padding:10px 12px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; font-size:12px; color:#334155; line-height:1.5;">
                        <b>Prescriptive Playbook:</b> {action}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning(f"Customer ID `{selected_cust}` not found in feature store.")

    # ==========================================================================
    # MODE B: WHAT-IF BEHAVIORAL SIMULATOR
    # ==========================================================================
    else:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Mode B: What-If Behavioral Feature Simulator</div>", unsafe_allow_html=True)
        st.caption("Adjust behavioral sliders across 13 engineered features to observe real-time model inferences.")

        col_s1, col_s2, col_s3 = st.columns(3, gap="medium")

        with col_s1:
            st.markdown("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Engagement & Tenure:</div>", unsafe_allow_html=True)
            sim_recency = st.slider("Recency (Days since last purchase):", 0, 365, 45)
            sim_tenure = st.slider("Tenure (Days since first order):", 0, 365, 180)
            sim_span = st.slider("Purchase Span (Days active):", 0, 365, 135)
            sim_freq = st.slider("Frequency (Distinct orders):", 1, 100, 4)

        with col_s2:
            st.markdown("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Spend & Basket Size:</div>", unsafe_allow_html=True)
            sim_monetary = st.slider("Monetary Total ($ Lifetime Spend):", 10.0, 30000.0, 1200.0, step=50.0)
            sim_aov = st.slider("Average Order Value ($):", 5.0, 5000.0, 300.0, step=25.0)
            sim_qty = st.slider("Total Units Purchased:", 1, 10000, 350)
            sim_basket = st.slider("Average Basket Size (Units):", 1.0, 500.0, 75.0)
            sim_unique = st.slider("Unique Products Purchased:", 1, 300, 25)

        with col_s3:
            st.markdown("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Risk Indicators & Demographics:</div>", unsafe_allow_html=True)
            sim_cancels = st.slider("Cancellation Count:", 0, 20, 0)
            sim_cancel_rate = st.slider("Cancellation Rate (0.0 - 1.0):", 0.0, 1.0, 0.0, step=0.05)
            sim_is_uk = st.selectbox("Customer Domicile:", ["UK Domestic (1)", "International Export (0)"])
            sim_repeat = st.selectbox("Repeat Customer Status:", ["Repeat (1)", "Single Order (0)"])

        features_input = {
            "recency_days": sim_recency,
            "frequency": sim_freq,
            "monetary_total": sim_monetary,
            "avg_order_value": sim_aov,
            "total_quantity": sim_qty,
            "avg_basket_size": sim_basket,
            "unique_products": sim_unique,
            "tenure_days": sim_tenure,
            "purchase_span_days": sim_span,
            "cancellation_count": sim_cancels,
            "cancellation_rate": sim_cancel_rate,
            "is_uk": 1 if "UK" in sim_is_uk else 0,
            "repeat_customer": 1 if "Repeat" in sim_repeat else 0
        }

        sim_res = predict_customer(features_input)
        s_prob = sim_res["churn_probability"]
        s_tier = sim_res["risk_tier"]
        s_badge = sim_res["badge_class"]
        s_action = sim_res["retention_playbook"]

        st.markdown(f"""
        <div class="panel-card" style="margin-top:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:14px; font-weight:700; color:#0f172a;">Simulation Inference Outcome</span>
                <span class="badge {s_badge}">{s_tier}</span>
            </div>
            <div style="margin-top:8px; display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:26px; font-weight:700; color:#2563eb;">{s_prob*100:.2f}%</span>
                <span style="font-size:12px; color:#64748b;">Predicted Probability of Churn</span>
            </div>
            <div style="margin-top:10px; padding:10px 12px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; font-size:12px; color:#334155; line-height:1.5;">
                <b>Tactical Action Recommendation:</b> {s_action}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # REST API Inspector
        with st.expander("REST API Payload Inspector"):
            api_col1, api_col2 = st.columns(2)
            with api_col1:
                st.caption("POST http://127.0.0.1:8000/predict")
                st.code(json.dumps(features_input, indent=2), language="json")
            with api_col2:
                st.caption("HTTP 200 OK Response")
                response_payload = {
                    "churn_probability": round(s_prob, 4),
                    "churn_prediction": sim_res["churn_prediction"],
                    "risk_tier": s_tier,
                    "retention_playbook": s_action,
                    "model_version": "v1.0-production"
                }
                st.code(json.dumps(response_payload, indent=2), language="json")
