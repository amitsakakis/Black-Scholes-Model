import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

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

class BinomialModel:
    def __init__(self, S, K, T, r, sigma, N):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N  # Number of steps
        
    def price(self, option_type):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        # Build binomial tree
        tree = np.zeros((self.N + 1, self.N + 1))
        
        # Calculate terminal payoffs
        for i in range(self.N + 1):
            ST = self.S * (u ** (self.N - i)) * (d ** i)
            if option_type == 'Call':
                tree[i, self.N] = max(0, ST - self.K)
            else:
                tree[i, self.N] = max(0, self.K - ST)
        
        # Backward induction
        for j in range(self.N - 1, -1, -1):
            for i in range(j + 1):
                tree[i, j] = np.exp(-self.r * dt) * (p * tree[i, j + 1] + (1 - p) * tree[i + 1, j + 1])
        
        return tree[0, 0], tree
    
    def get_stock_tree(self):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        for j in range(self.N + 1):
            for i in range(j + 1):
                stock_tree[i, j] = self.S * (u ** (j - i)) * (d ** i)
        
        return stock_tree

class MonteCarloModel:
    def __init__(self, S, K, T, r, sigma, N_simulations):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N_simulations = N_simulations
        
    def price(self, option_type):
        np.random.seed(42)
        Z = np.random.standard_normal(self.N_simulations)
        
        # Simulate terminal stock prices
        ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        # Calculate payoffs
        if option_type == 'Call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)
        
        # Discount to present value
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        return option_price, ST, payoffs
    
    def simulate_paths(self, n_paths=100, n_steps=100):
        """Simulate multiple price paths for visualization"""
        np.random.seed(42)
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S
        
        for i in range(n_paths):
            for t in range(1, n_steps + 1):
                Z = np.random.standard_normal()
                paths[i, t] = paths[i, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                                       self.sigma * np.sqrt(dt) * Z)
        
        return paths

def plot_binomial_tree(stock_tree, option_tree, N):
    """Visualize the binomial tree"""
    fig = go.Figure()
    
    # Only show first few steps if tree is large
    max_display = min(N, 8)
    
    # Add stock price nodes
    for j in range(max_display + 1):
        for i in range(j + 1):
            x_pos = j
            y_pos = j - 2*i
            
            stock_price = stock_tree[i, j]
            option_value = option_tree[i, j]
            
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[y_pos],
                mode='markers+text',
                marker=dict(size=20, color='lightblue', line=dict(color='blue', width=2)),
                text=f"S={stock_price:.2f}<br>V={option_value:.2f}",
                textposition="top center",
                textfont=dict(size=8),
                hovertext=f"Stock: ${stock_price:.2f}<br>Option: ${option_value:.2f}",
                showlegend=False
            ))
            
            # Add edges
            if j < max_display:
                # Up move
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos + 1],
                    y=[y_pos, y_pos + 1],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
                # Down move
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos + 1],
                    y=[y_pos, y_pos - 1],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
    
    fig.update_layout(
        title=f'Binomial Tree (First {max_display} steps shown)',
        xaxis=dict(title='Time Steps', showgrid=False),
        yaxis=dict(title='Price Levels', showgrid=False),
        height=500,
        showlegend=False
    )
    
    return fig

def plot_monte_carlo_paths(paths, K, option_type):
    """Visualize Monte Carlo simulation paths"""
    fig = go.Figure()
    
    n_paths, n_steps = paths.shape
    time_steps = np.linspace(0, 1, n_steps)
    
    # Plot paths (show subset if too many)
    display_paths = min(n_paths, 50)
    for i in range(display_paths):
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=paths[i],
            mode='lines',
            line=dict(width=1),
            opacity=0.3,
            showlegend=False
        ))
    
    # strike price line
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=[K] * n_steps,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'Strike Price (${K})'
    ))
    
    #mean path
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=mean_path,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Mean Path'
    ))
    
    fig.update_layout(
        title=f'Monte Carlo Simulation: {display_paths} Price Paths',
        xaxis_title='Time (Years)',
        yaxis_title='Stock Price ($)',
        height=500
    )
    
    return fig

def plot_monte_carlo_distribution(ST, K, option_type):
    """Show distribution of terminal stock prices"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ST,
        nbinsx=50,
        name='Terminal Prices',
        opacity=0.7
    ))
    
    # strike price line
    fig.add_shape(
        type="line",
        x0=K, x1=K, y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=3, dash="dash")
    )
    
    fig.add_annotation(
        x=K, y=0.9,
        yref="paper",
        text=f"Strike: ${K}",
        showarrow=True,
        arrowhead=2
    )
    
    if option_type == 'Call':
        itm_region = ST > K
        title_suffix = f"In-the-Money if > ${K}"
    else:
        itm_region = ST < K
        title_suffix = f"In-the-Money if < ${K}"
    
    fig.update_layout(
        title=f'Distribution of Terminal Stock Prices<br><sub>{title_suffix}</sub>',
        xaxis_title='Stock Price ($)',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig

def plot_convergence(S, K, T, r, sigma, option_type, max_steps=200):
    """Show how binomial and Monte Carlo converge to Black-Scholes"""
    bs_model = BlackScholesModel(S, K, T, r, sigma)
    bs_price = bs_model.price(option_type)
    
    # Binomial convergence
    steps = range(5, max_steps, 5)
    binomial_prices = []
    for n in steps:
        binom_model = BinomialModel(S, K, T, r, sigma, n)
        price, _ = binom_model.price(option_type)
        binomial_prices.append(price)
    
    # Monte Carlo convergence
    simulations = range(100, 10000, 100)
    mc_prices = []
    for n_sim in simulations:
        mc_model = MonteCarloModel(S, K, T, r, sigma, n_sim)
        price, _, _ = mc_model.price(option_type)
        mc_prices.append(price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(steps),
        y=binomial_prices,
        mode='lines+markers',
        name='Binomial Model',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(simulations),
        y=mc_prices,
        mode='lines+markers',
        name='Monte Carlo Model',
        line=dict(color='green'),
        xaxis='x2'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, max_steps],
        y=[bs_price, bs_price],
        mode='lines',
        name='Black-Scholes Price',
        line=dict(color='red', dash='dash', width=3)
    ))
    
    fig.update_layout(
        title='Model Convergence to Black-Scholes',
        xaxis=dict(title='Binomial Steps', domain=[0, 0.45]),
        xaxis2=dict(title='Monte Carlo Simulations', domain=[0.55, 1], anchor='y'),
        yaxis=dict(title='Option Price ($)'),
        height=400,
        showlegend=True
    )
    
    return fig

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
        legend=dict(x=0, y=1.1, orientation='h'),
        height=500
    )
    return fig

def main():
    st.set_page_config(page_title="Advanced Options Pricing Models", layout="wide")

    st.markdown("<h1 style='text-align: center;'>Advanced Options Pricing Models</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Akaash Mitsakakis-Nath</h2>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: center;'>
    Compare three methods for pricing European options: Black-Scholes (analytical), 
    Binomial Tree (discrete-time), and Monte Carlo (simulation-based)
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Option Parameters")
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        S = st.slider("Current Stock Price (S)", 50.0, 150.0, 100.0, 1.0, format="%.2f")
        K = st.slider("Strike Price (K)", 50.0, 150.0, 100.0, 1.0, format="%.2f")
        T = st.slider("Time to Maturity (T years)", 0.1, 2.0, 1.0, 0.1, format="%.2f")
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, 0.01, format="%.2f")
        sigma = st.slider("Volatility (σ)", 0.1, 0.5, 0.2, 0.01, format="%.2f")
        
        st.markdown("---")
        st.subheader("Model Settings")
        N_binomial = st.slider("Binomial Steps", 10, 200, 50, 10)
        N_monte_carlo = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, 1000)
    
    # Calculate prices with all three models
    bs_model = BlackScholesModel(S, K, T, r, sigma)
    bs_price = bs_model.price(option_type)
    bs_greeks = bs_model.greeks(option_type)
    
    binom_model = BinomialModel(S, K, T, r, sigma, N_binomial)
    binom_price, option_tree = binom_model.price(option_type)
    stock_tree = binom_model.get_stock_tree()
    
    mc_model = MonteCarloModel(S, K, T, r, sigma, N_monte_carlo)
    mc_price, ST, payoffs = mc_model.price(option_type)
    mc_paths = mc_model.simulate_paths()
    
    # Display prices comparison
    st.markdown("---")
    st.header("📊 Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Black-Scholes")
        st.markdown(f"<h2 style='color: green;'>${bs_price:.4f}</h2>", unsafe_allow_html=True)
        st.caption("Analytical solution")
        
    with col2:
        st.markdown("### Binomial Tree")
        st.markdown(f"<h2 style='color: blue;'>${binom_price:.4f}</h2>", unsafe_allow_html=True)
        diff_binom = ((binom_price - bs_price) / bs_price) * 100
        st.caption(f"Difference: {diff_binom:+.2f}%")
        
    with col3:
        st.markdown("### Monte Carlo")
        st.markdown(f"<h2 style='color: orange;'>${mc_price:.4f}</h2>", unsafe_allow_html=True)
        diff_mc = ((mc_price - bs_price) / bs_price) * 100
        st.caption(f"Difference: {diff_mc:+.2f}%")
    
    # Greeks (Black-Scholes only)
    st.markdown("---")
    st.header("📈 Option Greeks (Black-Scholes)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta (Δ)", f"{bs_greeks['delta']:.4f}")
    col2.metric("Gamma (Γ)", f"{bs_greeks['gamma']:.4f}")
    col3.metric("Theta (Θ)", f"{bs_greeks['theta']:.4f}")
    col4.metric("Vega (ν)", f"{bs_greeks['vega']:.4f}")
    col5.metric("Rho (ρ)", f"{bs_greeks['rho']:.4f}")
    
    # Tabs for different visualizations
    st.markdown("---")
    st.header("🔍 Model Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Binomial Tree", "Monte Carlo Simulation", "Convergence Analysis"])
    
    with tab1:
        st.markdown("### Binomial Tree Structure")
        st.markdown("""
        The binomial model builds a tree of possible stock prices, working backwards from expiration.
        Each node shows the stock price and option value at that point.
        """)
        fig_tree = plot_binomial_tree(stock_tree, option_tree, N_binomial)
        st.plotly_chart(fig_tree, use_container_width=True)
        
        st.info(f"""
        **How it works:**
        - Tree has {N_binomial} time steps
        - At each step, price can move up or down
        - Option value calculated by working backwards from terminal payoffs
        - Converges to Black-Scholes as steps increase
        """)
    
    with tab2:
        st.markdown("### Monte Carlo Price Paths")
        st.markdown("""
        Monte Carlo simulation generates thousands of random price paths following geometric Brownian motion.
        The option price is the average discounted payoff across all simulations.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_paths = plot_monte_carlo_paths(mc_paths, K, option_type)
            st.plotly_chart(fig_paths, use_container_width=True)
        
        with col2:
            fig_dist = plot_monte_carlo_distribution(ST, K, option_type)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        itm_percentage = (np.sum(payoffs > 0) / len(payoffs)) * 100
        avg_payoff = np.mean(payoffs[payoffs > 0]) if np.any(payoffs > 0) else 0
        
        st.info(f"""
        **Simulation Results:**
        - {N_monte_carlo:,} paths simulated
        - {itm_percentage:.1f}% finished in-the-money
        - Average ITM payoff: ${avg_payoff:.2f}
        - Discounted expected value: ${mc_price:.4f}
        """)
    
    with tab3:
        st.markdown("### Model Convergence")
        st.markdown("""
        As we increase the number of steps (Binomial) or simulations (Monte Carlo),
        both models converge to the Black-Scholes analytical solution.
        """)
        
        with st.spinner("Calculating convergence..."):
            fig_conv = plot_convergence(S, K, T, r, sigma, option_type)
            st.plotly_chart(fig_conv, use_container_width=True)
        
        st.success("""
        **Key Insights:**
        - Binomial converges smoothly with more steps
        - Monte Carlo shows more variation due to randomness
        - Both approach the Black-Scholes price as a limit
        """)
    
    # Broker Order Flow
    st.markdown("---")
    st.header("📉 Broker Order Flow Analysis")
    ticker = st.text_input("Enter Stock Ticker for Order Flow Analysis", value="AAPL")
    
    if ticker:
        try:
            fig_order_flow = plot_broker_order_flow(ticker)
            st.plotly_chart(fig_order_flow, use_container_width=True)
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}: {str(e)}")

if __name__ == "__main__":
    main()