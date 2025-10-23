#!/usr/bin/env python3
"""
ETF/Stock Portfolio Analyzer
Reads portfolio from CSV, calculates returns, and shows composition breakdown
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_portfolio(csv_path: str = "portfolio.csv") -> pd.DataFrame:
    """Load portfolio from CSV file."""
    df = pd.read_csv(csv_path)

    # Validate columns
    if not all(col in df.columns for col in ["ticker", "allocation"]):
        raise ValueError("CSV must have 'ticker' and 'allocation' columns")

    # Validate allocations
    total = df["allocation"].sum()
    if abs(total - 100) > 0.01:
        print(f"Warning: Allocations sum to {total}%, not 100%")

    return df


def get_historical_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical price data for a ticker."""
    print(f"Fetching price data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    return df


def calculate_returns(prices: pd.DataFrame) -> pd.Series:
    """Calculate cumulative returns from price data."""
    # Use adjusted close if available, otherwise close
    price_col = "Close"

    # Normalize to start at 0%
    initial_price = prices[price_col].iloc[0]
    returns = (prices[price_col] / initial_price - 1) * 100

    return returns


def get_etf_holdings(ticker: str, use_cache: bool = True) -> Dict[str, float]:
    """
    Fetch top holdings for an ETF and cache the results.
    Returns dict of {stock_ticker: weight_percentage}
    """
    cache_file = CACHE_DIR / f"{ticker}.json"

    # Check cache first
    if use_cache and cache_file.exists():
        print(f"Loading cached holdings for {ticker}")
        with open(cache_file, "r") as f:
            return json.load(f)

    print(f"Fetching ETF holdings for {ticker}...")

    try:
        etf = yf.Ticker(ticker)

        # Try to get holdings info
        info = etf.info

        # Check if this is actually an ETF/fund
        quote_type = info.get("quoteType", "")
        if quote_type not in ["ETF", "MUTUALFUND"]:
            # Not an ETF, return empty
            return {}

        # Get funds data which contains holdings
        try:
            funds_data = etf.funds_data
            if funds_data is None:
                return {}

            # Get top holdings (Symbol is index, "Holding Percent" is column)
            if not hasattr(funds_data, 'top_holdings'):
                return {}

            holdings_data = funds_data.top_holdings

            if holdings_data is None or holdings_data.empty:
                print(f"Warning: No holdings data available for {ticker}")
                return {}

            # Convert to dict: Symbol (index) -> Holding Percent * 100 (to get percentage)
            holdings_dict = {}

            for symbol, row in holdings_data.iterrows():
                if "Holding Percent" in row:
                    weight = row["Holding Percent"] * 100  # Convert decimal to percentage
                    if pd.notna(weight) and weight > 0:
                        holdings_dict[str(symbol)] = float(weight)

            if not holdings_dict:
                print(f"Warning: Could not parse holdings data for {ticker}")
                return {}

            # Cache the results
            with open(cache_file, "w") as f:
                json.dump(holdings_dict, f, indent=2)

            print(f"Cached {len(holdings_dict)} holdings for {ticker}")
            return holdings_dict

        except (AttributeError, Exception) as e:
            # funds_data not available or other error
            print(f"Could not fetch holdings: {e}")
            return {}

    except Exception as e:
        print(f"Error fetching holdings for {ticker}: {e}")
        return {}


def calculate_composition(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the true composition of individual stocks across the portfolio.
    Aggregates direct holdings and holdings within ETFs.
    """
    composition = {}

    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        allocation = row["allocation"]

        # Check if it's an ETF or individual stock
        # Simple heuristic: assume it's an ETF if it's in common ETF format
        # Better approach: check the asset type, but for MVP we'll try to fetch holdings
        holdings = get_etf_holdings(ticker)

        if holdings:
            # It's an ETF, add its underlying holdings
            print(f"{ticker} is an ETF with {len(holdings)} holdings")
            for stock, weight in holdings.items():
                # Weight in portfolio = ETF allocation * stock weight in ETF
                portfolio_weight = (allocation / 100) * weight
                composition[stock] = composition.get(stock, 0) + portfolio_weight
        else:
            # It's an individual stock
            print(f"{ticker} is an individual stock")
            composition[ticker] = composition.get(ticker, 0) + allocation

    # Convert to DataFrame and sort
    comp_df = pd.DataFrame([
        {"ticker": ticker, "composition": weight}
        for ticker, weight in composition.items()
    ]).sort_values("composition", ascending=False)

    return comp_df


def create_returns_chart(returns_data: Dict[str, pd.Series], portfolio: pd.DataFrame) -> go.Figure:
    """Create an interactive Plotly chart of returns."""
    fig = go.Figure()

    # Define colors (matching the reference image style)
    colors = ["#4285F4", "#EA4335", "#FBBC04", "#34A853", "#9C27B0", "#FF6F00"]

    for idx, (ticker, returns) in enumerate(returns_data.items()):
        allocation = portfolio[portfolio["ticker"] == ticker]["allocation"].values[0]
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns.values,
            mode="lines",
            name=f"{ticker} ({allocation}%)",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{ticker}</b><br>%{{y:.2f}}%<extra></extra>"
        ))

    # Update layout to match reference image style
    fig.update_layout(
        title="Total Return (%)",
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(
            ticksuffix="%",
            gridcolor="rgba(128,128,128,0.2)",
        ),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0
        ),
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    return fig


def print_composition_table(composition: pd.DataFrame):
    """Print the composition table to console."""
    print("\n" + "="*60)
    print("TOP HOLDINGS BY COMPOSITION")
    print("="*60)
    print(f"{'Ticker':<10} {'Composition':>15}")
    print("-"*60)

    for _, row in composition.head(20).iterrows():
        ticker = row["ticker"]
        comp = row["composition"]
        print(f"{ticker:<10} {comp:>14.2f}%")

    print("="*60)
    print(f"{'TOTAL':<10} {composition['composition'].sum():>14.2f}%")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze portfolio returns and composition"
    )
    parser.add_argument(
        "--portfolio",
        default="portfolio.csv",
        help="Path to portfolio CSV file (default: portfolio.csv)"
    )
    parser.add_argument(
        "--period",
        default="5y",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        help="Time period for historical data (default: 5y)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached ETF holdings data"
    )
    parser.add_argument(
        "--output",
        default="portfolio_analysis.html",
        help="Output HTML file for chart (default: portfolio_analysis.html)"
    )

    args = parser.parse_args()

    # Load portfolio
    print(f"Loading portfolio from {args.portfolio}...")
    portfolio = load_portfolio(args.portfolio)
    print(f"Loaded {len(portfolio)} positions")
    print(portfolio)
    print()

    # Fetch price data and calculate returns
    returns_data = {}
    for ticker in portfolio["ticker"]:
        try:
            prices = get_historical_prices(ticker, args.period)
            returns = calculate_returns(prices)
            returns_data[ticker] = returns
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if not returns_data:
        print("Error: No valid return data fetched")
        return

    print(f"\nSuccessfully fetched data for {len(returns_data)} tickers\n")

    # Calculate composition
    print("Calculating portfolio composition...")
    composition = calculate_composition(portfolio)

    # Display composition table
    print_composition_table(composition)

    # Create and save chart
    print(f"Creating returns chart...")
    fig = create_returns_chart(returns_data, portfolio)
    fig.write_html(args.output)
    print(f"Chart saved to {args.output}")

    # Also save as PNG for README
    png_output = args.output.replace('.html', '.png')
    fig.write_image(png_output, width=1200, height=600)
    print(f"Chart image saved to {png_output}")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(args.output)}")
    print(f"Opening chart in browser...")


if __name__ == "__main__":
    main()
