# ==============================================================================
# Project: Portfolio Optimization & Risk Analysis Tool
# Author: Lerato Mokoena
# Purpose: To construct an optimal investment portfolio that maximizes the 
#          Sharpe Ratio using Modern Portfolio Theory (MPT).
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: IMPORTS
# ------------------------------------------------------------------------------
# yfinance: Used to download real-time and historical stock market data.
# scipy.optimize: Provides mathematical optimization functions to find the best asset weights.
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ------------------------------------------------------------------------------
# SECTION 2: DATA ACQUISITION
# ------------------------------------------------------------------------------
def fetch_financial_data(tickers, start_date, end_date):
    """
    Downloads historical price data for a list of stocks.
    Handles both 'Close' and 'Adj Close' dynamically across different yfinance versions.
    """
    print(f"-> Fetching data for: {', '.join(tickers)}")
    df = yf.download(tickers, start=start_date, end=end_date)
    
    # Modern yfinance versions provide adjusted prices under 'Close'
    if 'Close' in df.columns:
        data = df['Close']
    elif 'Adj Close' in df.columns:
        data = df['Adj Close']
    else:
        data = df
        
    # Drop any missing values to ensure clean matrices for optimization
    return data.dropna()

# ------------------------------------------------------------------------------
# SECTION 3: PORTFOLIO MATHEMATICS
# ------------------------------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculates the expected annualized return and volatility (risk) of a portfolio.
    
    - Annualized Return: The weighted sum of individual asset returns, multiplied by 252 (trading days).
    - Annualized Volatility: Calculated using the covariance matrix to account for how 
      different stocks move in relation to one another.
    """
    # Calculate portfolio return
    returns = np.sum(mean_returns * weights) * 252
    
    # Calculate portfolio volatility (standard deviation)
    # This uses linear algebra (dot products) to factor in stock correlations
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Calculate Sharpe Ratio: measures excess return per unit of risk
    sharpe_ratio = (returns - risk_free_rate) / volatility
    
    return returns, volatility, sharpe_ratio

# ------------------------------------------------------------------------------
# SECTION 4: OPTIMIZATION ALGORITHM
# ------------------------------------------------------------------------------
def minimize_negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    The objective function for our optimizer. 
    
    SciPy only has 'minimize' functions, but we want to MAXIMIZE the Sharpe ratio.
    Therefore, we tell the algorithm to MINIMIZE the *negative* Sharpe ratio.
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Finds the optimal mathematical weighting of assets to maximize the Sharpe Ratio.
    """
    num_assets = len(mean_returns)
    
    # We start by guessing an equal weight for all assets (e.g., 25% each for 4 stocks)
    initial_guess = num_assets * [1. / num_assets]
    
    # Constraint: All asset weights must add up to 1.0 (100% of the portfolio)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: No short selling allowed. Weights must be between 0.0 (0%) and 1.0 (100%)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Run the SciPy Sequential Least Squares Programming (SLSQP) optimizer
    optimized_result = sco.minimize(minimize_negative_sharpe, initial_guess, 
                                    args=(mean_returns, cov_matrix, risk_free_rate), 
                                    method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimized_result

# ------------------------------------------------------------------------------
# SECTION 5: VISUALIZATION (The Efficient Frontier)
# ------------------------------------------------------------------------------
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios, optimal_weights):
    """
    Generates thousands of random portfolios to plot the "Efficient Frontier",
    and places a star on our mathematically optimal portfolio.
    """
    print("-> Simulating random portfolios to build the Efficient Frontier...")
    results = np.zeros((3, num_portfolios))
    num_assets = len(mean_returns)
    
    for i in range(num_portfolios):
        # Generate random weights and normalize them so they sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Get performance metrics for this random portfolio
        pret, pvol, psharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = pvol      # X-axis: Volatility (Risk)
        results[1,i] = pret      # Y-axis: Return
        results[2,i] = psharpe   # Color: Sharpe Ratio
        
    # Get the metrics for our actual OPTIMIZED portfolio
    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    
    # Mark the optimal portfolio with a red star
    plt.scatter(opt_vol, opt_ret, marker='*', color='r', s=500, label='Maximum Sharpe Portfolio')
    
    plt.title('Portfolio Optimization: The Efficient Frontier')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing=0.8)
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300)
    print("-> Visualization saved: 'efficient_frontier.png'")
    plt.close()

# ------------------------------------------------------------------------------
# SECTION 6: MAIN EXECUTION PIPELINE
# ------------------------------------------------------------------------------
def run_optimization():
    # 1. Define our asset basket (Tech, Finance, Consumer Goods, and Healthcare)
    tickers = ['AAPL', 'JPM', 'KO', 'JNJ']
    
    # 2. Fetch the last 3 years of data
    data = fetch_financial_data(tickers, start_date='2021-01-01', end_date='2024-01-01')
    
    # 3. Calculate daily returns
    returns = data.pct_change()
    
    # 4. Calculate the average daily returns and the covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    print("\n-> Running mathematical optimization...")
    # 5. Run the optimizer to find the best weights
    optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)
    optimal_weights = optimal_portfolio.x
    
    print("\n--- OPTIMAL PORTFOLIO ALLOCATION ---")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight*100:.2f}%")
        
    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    print(f"\nExpected Annual Return: {opt_ret*100:.2f}%")
    print(f"Annual Volatility (Risk): {opt_vol*100:.2f}%")
    print(f"Sharpe Ratio: {opt_sharpe:.2f}\n")
    
    # 6. Generate the graph
    plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, optimal_weights=optimal_weights)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    run_optimization()
