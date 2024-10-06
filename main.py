import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf  # Added for pulling stock data

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility

    def calculate_d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type):
        d1, d2 = self.calculate_d1_d2()
        if option_type == 'Call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'Put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError('Option type not recognized')

    def greeks(self, option_type):
        d1, d2 = self.calculate_d1_d2()
        delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) if option_type == 'Call' else (
                -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) if option_type == 'Call' else (
              -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2))
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="1y")
    return hist

def plot_broker_order_flow(ticker):
    hist = fetch_stock_data(ticker)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', yaxis='y2', opacity=0.5))
    
    fig.update_layout(
        title=f'Order Flow Analysis for {ticker}',
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        xaxis_title='Date',
        legend=dict(x=0, y=1.1, orientation='h')
    )
    return fig

def plot_option_prices(variable, values, S, K, T, r, sigma, option_type):
    if variable == "Stock Price(S)":
        prices = [BlackScholesModel(val, K, T, r, sigma).price(option_type) for val in values]
    elif variable == "Strike Price(K)":
        prices = [BlackScholesModel(S, val, T, r, sigma).price(option_type) for val in values]
    elif variable == "Time to Maturity(T)":
        prices = [BlackScholesModel(S, K, val, r, sigma).price(option_type) for val in values]
    elif variable == "Risk-Free Interest Rate(r)":
        prices = [BlackScholesModel(S, K, T, val, sigma).price(option_type) for val in values]
    elif variable == "Volatility(σ)":
        prices = [BlackScholesModel(S, K, T, r, val).price(option_type) for val in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values, y=prices, mode='lines', name=f'{option_type.capitalize()} Option Price'))
    fig.update_layout(title=f'{option_type.capitalize()} Option Prices vs. {variable.capitalize()} ',
                      xaxis_title=variable.capitalize(),
                      yaxis_title='Option Price')
    return fig

def plot_greeks(variable, values, S, K, T, r, sigma, option_type):
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    fig = go.Figure()
    
    for greek in greeks:
        if variable == "Stock Price(S)":
            greek_values = [BlackScholesModel(val, K, T, r, sigma).greeks(option_type)[greek] for val in values]
        elif variable == "Strike Price(K)":
            greek_values = [BlackScholesModel(S, val, T, r, sigma).greeks(option_type)[greek] for val in values]
        elif variable == "Time to Maturity(T)":
            greek_values = [BlackScholesModel(S, K, val, r, sigma).greeks(option_type)[greek] for val in values]
        elif variable == "Risk-Free Interest Rate(r)":
            greek_values = [BlackScholesModel(S, K, T, val, sigma).greeks(option_type)[greek] for val in values]
        elif variable == "Volatility(σ)":
            greek_values = [BlackScholesModel(S, K, T, r, val).greeks(option_type)[greek] for val in values]

        fig.add_trace(go.Scatter(x=values, y=greek_values, mode='lines', name=f'{greek.capitalize()} ({option_type.capitalize()})'))

    fig.update_layout(title=f'{option_type.capitalize()} Option Greeks vs. {variable.capitalize()}',
                      xaxis_title=variable.capitalize(),
                      yaxis_title='Value')
    return fig

def main():
    st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")

    st.markdown("<h1 style='text-align: center;'>Black-Scholes Option Pricing Model</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Akaash Mitsakakis-Nath</h2>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center;'>This app calculates the price and greeks of a European call or put option using the Black-Scholes model, as well as visually represents the results in a interactive plot.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: left; color: White;'>Option Type</h3>", unsafe_allow_html=True)
        option_type = st.selectbox("", ["Call", "Put"])
        st.subheader("Define Option Parameters")
        S = st.slider("Current Stock Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
        K = st.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
        T = st.slider("Time to Maturity (T)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.2f")
        r = st.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.01, format="%.2f")
        sigma = st.slider("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2, step=0.01, format="%.2f")

    model = BlackScholesModel(S, K, T, r, sigma)
    price = model.price(option_type)
    greeks = model.greeks(option_type)

    with col1: 
        st.markdown(f"<h2 style='text-align: left; color: Green;'>{option_type.capitalize()} Option Price: {price:.2f}</h2>", unsafe_allow_html=True)
        st.subheader("**Greeks:**")
        st.markdown(f"**Delta:** {greeks['delta']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {greeks['gamma']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {greeks['theta']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Vega:** {greeks['vega']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {greeks['rho']:.2f}", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: left; color: White;'>Select Independent Parameter</h3>", unsafe_allow_html=True)
        variable_to_plot = st.selectbox("", ["Stock Price(S)", "Strike Price(K)", "Time to Maturity(T)", "Risk-Free Interest Rate(r)", "Volatility(σ)"])
        
    if variable_to_plot == "Stock Price(S)":
        values = np.linspace(50, 150, 100)
    elif variable_to_plot == "Strike Price(K)":
        values = np.linspace(50, 150, 100)
    elif variable_to_plot == "Time to Maturity(T)":
        values = np.linspace(0.1, 2, 100)
    elif variable_to_plot == "Risk-Free Interest Rate(r)":
        values = np.linspace(0.0, 0.1, 100)
    elif variable_to_plot == "Volatility(σ)":
        values = np.linspace(0.1, 0.5, 100)

    with col2:
        fig_prices = plot_option_prices(variable_to_plot, values, S, K, T, r, sigma, option_type)
        st.plotly_chart(fig_prices)

        fig_greeks = plot_greeks(variable_to_plot, values, S, K, T, r, sigma, option_type)
        st.plotly_chart(fig_greeks)

    st.markdown("<h2 style='text-align: left;'>Broker Order Flow Analysis</h2>", unsafe_allow_html=True)
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    fig_order_flow = plot_broker_order_flow(ticker)
    st.plotly_chart(fig_order_flow)

if __name__ == "__main__":
    main()
