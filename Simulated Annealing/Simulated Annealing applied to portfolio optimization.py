#!/usr/bin/env python
"""
Simulated Annealing for Portfolio Optimization

This script uses simulated annealing to optimize a portfolio by maximizing the Sharpe Ratio.
It downloads historical stock data using yfinance, calculates annualized returns and covariance,
and then finds the optimal asset weights using simulated annealing.

If you have your own dataset (e.g., a CSV file with price data), replace the data loading section
with your code to read and process your dataset accordingly.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime


class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.01,
                 initial_temp=1000, cooling_rate=0.995, max_iter=10000, seed=None):
        """
        Initializes the PortfolioOptimizer.

        Parameters:
            expected_returns (np.array): Expected annualized returns for each asset.
            cov_matrix (np.array): Annualized covariance matrix of asset returns.
            risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.
            initial_temp (float): Starting temperature for simulated annealing.
            cooling_rate (float): Rate at which the temperature decreases.
            max_iter (int): Maximum number of iterations for the algorithm.
            seed (int): Random seed for reproducibility.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def portfolio_performance(self, weights):
        """
        Calculates portfolio performance metrics.

        Parameters:
            weights (np.array): Portfolio weights for each asset.

        Returns:
            tuple: (portfolio_return, portfolio_std, sharpe_ratio)
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    def fitness(self, weights):
        """
        Defines the fitness function to be minimized.
        We use the negative Sharpe ratio because simulated annealing is a minimization algorithm.
        """
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe

    def random_neighbor(self, weights, perturbation_scale=0.05):
        """
        Generates a neighboring solution by perturbing one of the weights.

        Parameters:
            weights (np.array): Current portfolio weights.
            perturbation_scale (float): Maximum change applied to a randomly selected weight.

        Returns:
            np.array: New portfolio weights (normalized to sum to 1).
        """
        new_weights = weights.copy()
        index = np.random.randint(len(weights))
        # Apply a random change that can be positive or negative
        change = np.random.uniform(-perturbation_scale, perturbation_scale)
        new_weights[index] += change
        # Ensure weights are non-negative
        new_weights = np.maximum(new_weights, 0)
        # Re-normalize to ensure full allocation
        if new_weights.sum() == 0:
            new_weights = np.ones_like(new_weights)
        new_weights /= new_weights.sum()
        return new_weights

    def simulated_annealing(self):
        """
        Core simulated annealing loop. It starts with a random portfolio allocation,
        then iteratively explores neighboring allocations, accepting worse solutions
        with a probability that decreases with the temperature.

        Returns:
            tuple: (best_weights, best_sharpe_ratio)
        """
        n_assets = len(self.expected_returns)
        # Start with random weights from a Dirichlet distribution
        current_weights = np.random.dirichlet(np.ones(n_assets), size=1)[0]
        current_fitness = self.fitness(current_weights)
        best_weights = current_weights.copy()
        best_fitness = current_fitness

        temperature = self.initial_temp

        for i in range(self.max_iter):
            candidate_weights = self.random_neighbor(current_weights)
            candidate_fitness = self.fitness(candidate_weights)
            delta = candidate_fitness - current_fitness

            # Accept candidate if it's better or with a probability based on temperature
            if delta < 0 or np.random.rand() < math.exp(-delta / temperature):
                current_weights = candidate_weights
                current_fitness = candidate_fitness
                if current_fitness < best_fitness:
                    best_weights = current_weights.copy()
                    best_fitness = current_fitness

            # Decrease temperature gradually
            temperature *= self.cooling_rate

            # Print progress periodically
            if (i + 1) % (self.max_iter // 10) == 0:
                print(
                    f"Iteration {i + 1}/{self.max_iter} | Best Sharpe Ratio: {-best_fitness:.4f} | Temperature: {temperature:.4f}")

            # Early stopping if temperature is sufficiently low
            if temperature < 1e-8:
                break

        return best_weights, -best_fitness  # Return best weights and positive Sharpe ratio

    def optimize(self):
        """
        A simple interface to run the simulated annealing optimization.

        Returns:
            tuple: (optimized weights, optimized Sharpe ratio)
        """
        return self.simulated_annealing()


def download_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted close price data for the given tickers.

    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data


def calculate_annualized_returns(data):
    """
    Calculates annualized returns based on daily returns.

    Parameters:
        data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.Series: Annualized returns for each asset.
    """
    daily_returns = data.pct_change().dropna()
    annualized_returns = daily_returns.mean() * 252  # 252 trading days per year
    return annualized_returns


def calculate_annualized_covariance(data):
    """
    Calculates the annualized covariance matrix of asset returns.

    Parameters:
        data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.DataFrame: Annualized covariance matrix.
    """
    daily_returns = data.pct_change().dropna()
    annualized_cov = daily_returns.cov() * 252
    return annualized_cov


def plot_cumulative_returns(data, portfolio_weights, tickers, benchmark_ticker="^GSPC", start_date=None, end_date=None):
    """
    Plots the cumulative returns of the optimized portfolio against a benchmark.

    Parameters:
        data (pd.DataFrame): DataFrame of adjusted close prices for the portfolio assets.
        portfolio_weights (np.array): Optimized portfolio weights.
        tickers (list): List of portfolio asset tickers.
        benchmark_ticker (str): Ticker symbol for the benchmark (default S&P 500).
        start_date (str): Start date for benchmark data.
        end_date (str): End date for benchmark data.
    """
    daily_returns = data.pct_change().dropna()
    portfolio_daily_returns = daily_returns.dot(portfolio_weights)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()

    # Download benchmark data for comparison
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_daily_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative = (1 + benchmark_daily_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative, label="Optimized Portfolio", lw=2)
    plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=benchmark_ticker, lw=2)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title("Optimized Portfolio vs Benchmark")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Define the list of assets (tickers) for portfolio optimization.
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Define the time period for historical data.
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    # Download historical price data.
    data = download_data(tickers, start_date, end_date)

    # Calculate the expected annualized returns and covariance matrix.
    expected_returns = calculate_annualized_returns(data).values
    cov_matrix = calculate_annualized_covariance(data).values

    # Initialize the PortfolioOptimizer with chosen hyperparameters.
    optimizer = PortfolioOptimizer(
        expected_returns,
        cov_matrix,
        risk_free_rate=0.01,
        initial_temp=1000,
        cooling_rate=0.995,
        max_iter=10000,
        seed=42
    )

    # Run the simulated annealing optimization.
    best_weights, best_sharpe = optimizer.optimize()

    print("\nOptimized Portfolio Weights:")
    for ticker, weight in zip(tickers, best_weights):
        print(f"{ticker}: {weight:.2%}")

    print(f"\nOptimized Portfolio Sharpe Ratio: {best_sharpe:.4f}")

    # Plot cumulative returns of the optimized portfolio versus a benchmark (S&P 500).
    plot_cumulative_returns(data, best_weights, tickers, benchmark_ticker="^GSPC", start_date=start_date,
                            end_date=end_date)
