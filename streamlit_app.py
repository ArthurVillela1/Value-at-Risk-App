import streamlit as st
import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import norm # type: ignore
import logging
import socket

st.set_page_config(layout="wide")
st.title("Value at Risk (VaR) Calculator")

with st.sidebar:
    st.write("`Created by: Arthur Villela`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    github_url = "https://github.com/ArthurVillela1"
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;">'
        f'<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">'
        f'</a><a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;">'
        f'<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"></a>',
        unsafe_allow_html=True
    )
    st.sidebar.write("--------------------------")
    portfolio_val = st.number_input('Portfolio Value (USD)', value=100000)
    tickers = st.text_input('Stock Tickers (e.g., META NVDA)', 'META NVDA')
    weights = st.text_input('Stock Weights (%):', '50 50')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))
    confidence_lv = st.slider('Confidence Level', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    rolling_window = st.slider('Rolling window', min_value=1, max_value=252, value=20)

# Normalize and prepare input data
tickers_list = [ticker.upper() for ticker in tickers.split()]
weights_list = list(map(float, weights.split()))
weights_list = [w / 100 for w in weights_list]
weights_array = np.array(weights_list)

# Error handling for mismatched weights and tickers
if len(weights_list) != len(tickers_list):
    st.error("The number of weights must match the number of tickers. Please adjust your inputs.")
    st.stop()

var_method = st.selectbox("Select VaR Method", ["Historical", "Parametric", "Monte Carlo Simulations"])

# Check network access
try:
    socket.create_connection(("www.google.com", 80))
    st.write("Network access: OK")
except OSError:
    st.error("Network access: FAILED")
    st.stop()

# Display library versions
st.write(f"yfinance version: {yf.__version__}")
st.write(f"pandas version: {pd.__version__}")
st.write(f"numpy version: {np.__version__}")
st.write(f"streamlit version: {st.__version__}")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch adjusted close data
adj_close_df = pd.DataFrame()
for ticker in tickers_list:
    try:
        logging.debug(f"Fetching data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
        logging.debug(f"Data for {ticker}: {data.head()}")
        if 'Adj Close' in data.columns and not data['Adj Close'].empty:
            adj_close_df[ticker] = data['Adj Close']
        else:
            st.warning(f"No data or 'Adj Close' column found for {ticker}. Skipping.")
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        logging.error(f"Error fetching data for {ticker}: {e}")

if adj_close_df.empty:
    st.error("No valid data found for the provided tickers. Please check your inputs.")
    st.stop()

# Calculate log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# Calculate portfolio returns based on weights
portfolio_returns = (log_returns * weights_list).sum(axis=1)

# Calculate mean returns and covariance matrix for assets
mean_returns = log_returns.mean().values
cov_matrix = log_returns.cov().values

# Function to calculate VaR with Monte Carlo Simulations
def monte_carlo_var_cov(simulations, mean_returns, cov_matrix, weights, portfolio_value, confidence_level):
    simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, simulations)
    portfolio_simulated_returns = np.dot(simulated_returns, weights)
    losses = portfolio_value * portfolio_simulated_returns 
    var = np.percentile(losses, 100 * (1 - confidence_level)) 
    return var, losses

# Function to calculate Parametric VaR
def parametric_var(portfolio_returns, confidence_level, portfolio_value):
    mean = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)
    z_score = norm.ppf(1 - confidence_level)
    var = (mean + z_score * sigma) * portfolio_value
    losses = portfolio_returns * portfolio_value
    return var, losses

# Function to calculate Historical VaR
def historical_var(portfolio_returns, confidence_level, portfolio_value):
    losses = portfolio_returns * portfolio_value 
    var = np.percentile(losses, 100 * (1 - confidence_level)) 
    return var, losses

# General VaR calculation function
def var_calculation(confidence_level, method, portfolio_value, simulations=None, mean_returns=None, cov_matrix=None, weights=None, portfolio_returns=None):
    if method == "Historical":
        return historical_var(portfolio_returns, confidence_level, portfolio_value)
    elif method == "Parametric":
        return parametric_var(portfolio_returns, confidence_level, portfolio_value)
    elif method == "Monte Carlo Simulations" and simulations is not None:
        if mean_returns is not None and cov_matrix is not None and weights is not None:
            return monte_carlo_var_cov(simulations, mean_returns, cov_matrix, weights, portfolio_value, confidence_level)
    return None, None

# Plot histogram function
def plot_histogram(losses, var_value):
    plt.figure(figsize=(8, 4))
    plt.hist(losses, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=var_value, color='r', linestyle='--', label=f'VaR: ${round(var_value, 2)}')
    plt.xlabel('Portfolio Variation (USD)')
    plt.ylabel('Frequency')
    plt.title('Portfolio Variation Distribution with VaR')
    plt.legend()
    st.pyplot(plt)

# Display VaR result and plot histogram
if var_method == "Monte Carlo Simulations":
    st.subheader(f"{var_method} Value at Risk for your portfolio at {int(confidence_lv * 100)}% confidence level:")
    var, losses = var_calculation(confidence_lv, var_method, portfolio_val, simulations=100000, 
                                  mean_returns=mean_returns, cov_matrix=cov_matrix, weights=weights_array)
    st.title(f"${round(var, 2)}")
    plot_histogram(losses, var)
else:
    st.subheader(f"{var_method} Value at Risk for your portfolio at {int(confidence_lv * 100)}% confidence level:")
    var, losses = var_calculation(confidence_lv, var_method, portfolio_val, portfolio_returns=portfolio_returns)
    st.title(f"${round(var, 2)}")
    plot_histogram(losses, var)