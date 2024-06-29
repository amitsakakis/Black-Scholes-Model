import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objs as go

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        # Initialize the Black-Scholes Model parameters
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility

    def calculate_d1_d2(self):
        # Calculate d1 and d2 using the Black-Scholes formula
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type):
        # Calculate the price of the option (call or put) using the Black-Scholes formula
        d1, d2 = self.calculate_d1_d2()
        if option_type == 'Call':
            # Price of a call option
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'Put':
            # Price of a put option
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError('Option type not recognized')

    def greeks(self, option_type):
        # Calculate the Greeks of the option (call or put)
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
        # Return the Greeks as a dictionary
        return {
            'delta': delta, 'gamma': gamma,
            'theta': theta, 'vega': vega,
            'rho': rho
        }

def plot_option_prices(variable, values, S, K, T, r, sigma, option_type):
    # Generate option prices for a range of values of a selected variable
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

    # Create a plotly figure to display option prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values, y=prices, mode='lines', name=f'{option_type.capitalize()} Option Price'))
    fig.update_layout(title=f'{option_type.capitalize()} Option Prices vs. {variable.capitalize()} ',
                      xaxis_title=variable.capitalize(),
                      yaxis_title='Option Price')
    return fig

def plot_greeks(variable, values, S, K, T, r, sigma, option_type):
    # Generate Greeks for a range of values of a selected variable
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

        # Add each Greek to the plot
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
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: left; color: White;'>Option Type</h3>", unsafe_allow_html=True)
        option_type = st.selectbox("", ["Call", "Put"])
        st.subheader("Define Option Parameters")
        # Define sliders for input parameters
        S = st.slider("Current Stock Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
        K = st.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
        T = st.slider("Time to Maturity (T)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.2f")
        r = st.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.01, format="%.2f")
        sigma = st.slider("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2, step=0.01, format="%.2f")

    # Create an instance of BlackScholesModel with the selected parameters
    model = BlackScholesModel(S, K, T, r, sigma)
    price = model.price(option_type)
    greeks = model.greeks(option_type)

    with col1: 
        st.markdown(f"<h2 style='text-align: left; color: Green;'>{option_type.capitalize()} Option Price: {price:.2f}</h2>", unsafe_allow_html=True)
        st.subheader("**Greeks:**")
        # Display the calculated Greeks
        st.markdown(f"**Delta:** {greeks['delta']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Gamma:** {greeks['gamma']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Theta:** {greeks['theta']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Vega:** {greeks['vega']:.2f}", unsafe_allow_html=True)
        st.markdown(f"**Rho:** {greeks['rho']:.2f}", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: left; color: White;'>Select Independent Parameter</h3>", unsafe_allow_html=True)
        variable_to_plot = st.selectbox("", ["Stock Price(S)", "Strike Price(K)", "Time to Maturity(T)", "Risk-Free Interest Rate(r)", "Volatility(σ)"])
        
    # Define the range of values for the selected variable
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
        # Plot the option prices and Greeks
        fig_prices = plot_option_prices(variable_to_plot, values, S, K, T, r, sigma, option_type)
        st.plotly_chart(fig_prices)

        fig_greeks = plot_greeks(variable_to_plot, values, S, K, T, r, sigma, option_type)
        st.plotly_chart(fig_greeks)

if __name__ == "__main__":
    main()
