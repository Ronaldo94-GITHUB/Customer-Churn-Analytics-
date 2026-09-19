import streamlit as st

from customer_churn.dashboard.churn_view import render_churn_dashboard

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")
st.title("Customer Churn Analytics")
st.caption("Produto independente · versão 1.0")
st.markdown("<style>.section-title {font-size:1.5rem;font-weight:600;margin:1.6rem 0 .8rem}</style>",
            unsafe_allow_html=True)
render_churn_dashboard()
