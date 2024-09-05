import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math
import pandas as pd
import yfinance as yf
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Value at Risk (VaR) Calculator")

with st.sidebar:
    st.write("`Created by: Arthur Villela`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    github_url ="https://github.com/ArthurVillela1"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"><a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"></a>', unsafe_allow_html=True)
    st.sidebar.write("--------------------------")
    portfolio_val = st.number_input('Portfolio Value (USD)', value=100000)
    tickers = st.text_input('Stock Tickers', 'META NVDA')
    weights = st.text_input('Stock Weights (%):', '20 80')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))
    confidence_lv = st.slider('Confidence Level', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    calculate_btn = st.button('Calculate VaR')

tickers_list = tickers.split(" ")
weights_list = tickers.split(" ")

var_method = st.selectbox("Select VaR Method", ["Historical", "Parametric", "Monte Carlo Simulations"])

def var_calculation(returns, confidence_level, method, portfolio_value):
    if method == "Historical":
        var = np.percentile(returns, 100 - confidence_level)*portfolio_value
    elif method == "Parametric":
        mean = np.mean(returns)
        sigma = np.std(returns)
        z_score = -(stats.norm.ppf(1 - confidence_level / 100))
        var = -(mean + z_score * sigma)*portfolio_value
    elif method == "Monte Carlo":
        var = ""
    return var

st.subheader(f"{var_method} Value at Risk for your portfolio at {int(confidence_lv*100)}% confidence level:")
#st.title(f":blue-background[{var_calculation(stock_returns, confidence_lv, var_method, portfolio_val)}]")