# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 07:07:40 2025

@author: Hemal
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from scipy.optimize import minimize

def fetch_data(tickers, start_date, end_date):
    data_frames = []
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
        data_frames.append(stock_data)

    data = pd.concat(data_frames, axis=1, keys=tickers)
    data.columns = tickers
    return data

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    portfolio_return = returns.mean().mean()
    portfolio_volatility = returns.std().mean()
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    portfolio_return = returns.mean().mean()
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std().mean()
    return (portfolio_return - risk_free_rate) / downside_deviation

def portfolio_optimization(returns):
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))

    def negative_sharpe_ratio(weights):
        return -(np.sum(returns.mean() * weights) - 0.02) / portfolio_volatility(weights)

    # Constraints to ensure diversification
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 1) for _ in range(returns.shape[1]))  # Minimum weight of 5%
    result = minimize(negative_sharpe_ratio, len(returns.columns) * [1. / len(returns.columns),],
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed.")

def main():
    st.title("Financial Analysis Web App")

    tickers = st.text_input("Enter stock tickers (comma separated)", "AAPL, MSFT, GOOG").split(",")
    tickers = [ticker.strip() for ticker in tickers]
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

    stock_data = fetch_data(tickers, start_date, end_date)
    returns_data = stock_data.pct_change().dropna()

    st.subheader("Stock Prices")
    st.line_chart(stock_data)

    st.subheader("Stock Returns")
    st.line_chart(returns_data)

    sharpe_ratio = calculate_sharpe_ratio(returns_data)
    sortino_ratio = calculate_sortino_ratio(returns_data)

    st.subheader("Performance Metrics")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Sortino Ratio: {sortino_ratio:.2f}")

    optimal_weights = portfolio_optimization(returns_data)
    st.subheader("Portfolio Optimization")
    st.write("Optimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        st.write(f"{ticker}: {weight:.2%}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(tickers, optimal_weights)
    ax.set_title("Optimal Portfolio Weights")
    ax.set_xlabel("Tickers")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

    csv_data = stock_data.to_csv()
    st.download_button("Download Stock Data CSV", csv_data, "stock_data.csv")

if __name__ == "__main__":
    main()
