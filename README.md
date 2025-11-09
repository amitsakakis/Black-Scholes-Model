<h1 align="center">Advanced Options Pricing Models</h1>

## Project Description

This web app prices European call and put options using three different models — Black-Scholes, Binomial Tree, and Monte Carlo. It also calculates Greeks and includes visualizations for each model. Built in Python with Streamlit.

Live app: https://blackscholeswithgreeks.streamlit.app

## Models

### Black-Scholes Model
A closed-form solution for pricing European options. It uses inputs such as stock price, strike price, time to expiry, volatility and the risk-free rate.

### Binomial Tree Model
A discrete-time approach. The stock price moves up or down at each step, and the option value is calculated backward through the tree. With more steps, it approaches the Black-Scholes price.

### Monte Carlo Simulation
Simulates a large number of possible future stock price paths. The option price is the discounted average of the payoffs across all simulations.

## Features

- Pricing using Black-Scholes, Binomial, and Monte Carlo models  
- Calculation of Delta, Gamma, Theta, Vega and Rho  
- Binomial tree visualization  
- Monte Carlo price paths and payoff distribution  
- Convergence of Binomial and Monte Carlo to Black-Scholes  
- Basic order flow chart using historical price and volume data

## Option Greeks

- **Delta** – Sensitivity of option price to stock price  
- **Gamma** – Rate of change of Delta  
- **Theta** – Time decay of the option’s value  
- **Vega** – Sensitivity to changes in volatility  
- **Rho** – Sensitivity to changes in interest rate

## Assumptions (Black-Scholes)

- European options only (no early exercise)  
- No dividends paid during the contract  
- No transaction costs or taxes  
- Constant risk-free rate and volatility  
- Stock returns are log-normally distributed  
- Markets are efficient

## License

© 2024 Akaash Mitsakakis-Nath  
MIT License
