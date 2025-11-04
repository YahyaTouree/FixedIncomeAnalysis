"""
Options Pricing & Volatility Analysis Dashboard

Interactive Streamlit application for real-time options pricing,
volatility analysis, and trading strategy evaluation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from black_scholes import BlackScholesMerton
from binomial_tree import BinomialTree

# Configure Streamlit page
st.set_page_config(
    page_title="Options Pricing & Volatility Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.greek-card {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 0.25rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_stock_data(ticker, period="1y"):
    """Fetch stock data with caching."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

@st.cache_data
def get_options_chain(ticker):
    """Fetch options chain data."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            return None, None, None

        # Get the nearest expiration
        nearest_exp = expirations[0]
        opt = stock.option_chain(nearest_exp)

        return opt.calls, opt.puts, nearest_exp
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None, None, None

def calculate_option_metrics(S, K, T, r, sigma, option_type='call'):
    """Calculate comprehensive option metrics."""
    bsm = BlackScholesMerton()
    bt = BinomialTree()

    # Black-Scholes pricing and Greeks
    if option_type == 'call':
        bs_price = bsm.call_price(S, K, T, r, sigma)
        bs_greeks = bsm.calculate_greeks(S, K, T, r, sigma)
    else:
        bs_price = bsm.put_price(S, K, T, r, sigma)
        bs_greeks = bsm.calculate_greeks(S, K, T, r, sigma)

    # Binomial tree pricing for comparison
    n_steps = 100
    if option_type == 'call':
        bt_euro = bt.european_call(S, K, T, r, sigma, n_steps)
        bt_amer = bt.american_call(S, K, T, r, sigma, n_steps)
    else:
        bt_euro = bt.european_put(S, K, T, r, sigma, n_steps)
        bt_amer = bt.american_put(S, K, T, r, sigma, n_steps)

    return {
        'black_scholes': {'price': bs_price, 'greeks': bs_greeks},
        'binomial_european': bt_euro,
        'binomial_american': bt_amer
    }

def create_price_chart(S, strikes, call_prices, put_prices, K_selected):
    """Create interactive price chart."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Options', 'Put Options'),
        horizontal_spacing=0.1
    )

    # Call prices
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=call_prices,
            mode='lines+markers',
            name='Call Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Highlight selected strike
    idx = np.argmin(np.abs(strikes - K_selected))
    fig.add_trace(
        go.Scatter(
            x=[K_selected],
            y=[call_prices[idx]],
            mode='markers',
            name='Selected Strike',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )

    # Put prices
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=put_prices,
            mode='lines+markers',
            name='Put Price',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    # Highlight selected strike
    fig.add_trace(
        go.Scatter(
            x=[K_selected],
            y=[put_prices[idx]],
            mode='markers',
            name='Selected Strike',
            marker=dict(color='blue', size=10),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Option Prices vs Strike Price",
        height=400,
        showlegend=True
    )

    fig.update_xaxes(title_text="Strike Price")
    fig.update_yaxes(title_text="Option Price")

    return fig

def create_greeks_heatmap(greeks_dict):
    """Create Greeks heatmap visualization."""
    greeks_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    call_values = [greeks_dict['delta_call'], greeks_dict['gamma'],
                   greeks_dict['vega'], greeks_dict['theta_call'], greeks_dict['rho_call']]
    put_values = [greeks_dict['delta_put'], greeks_dict['gamma'],
                  greeks_dict['vega'], greeks_dict['theta_put'], greeks_dict['rho_put']]

    fig = go.Figure(data=go.Heatmap(
        z=[call_values, put_values],
        x=greeks_names,
        y=['Call', 'Put'],
        colorscale='RdBu',
        zmid=0,
        text=[[f"{v:.4f}" for v in call_values],
              [f"{v:.4f}" for v in put_values]],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))

    fig.update_layout(
        title="Greeks Heatmap",
        height=300
    )

    return fig

def main():
    """Main application function."""
    st.title("📊 Options Pricing & Volatility Analysis")
    st.markdown("Advanced options pricing models with real-time market data")

    # Sidebar for inputs
    st.sidebar.header("Model Parameters")

    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

    # Option parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        strike_price = st.number_input("Strike Price", value=150.0, min_value=0.0)

    with col2:
        days_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
        volatility = st.number_input("Volatility (%)", value=25.0, min_value=1.0, max_value=200.0) / 100

    # Market parameters
    st.sidebar.subheader("Market Parameters")
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0) / 100
    dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=10.0) / 100

    # Fetch market data
    with st.spinner(f"Fetching data for {ticker}..."):
        stock_data, stock_info = get_stock_data(ticker)
        options_calls, options_puts, expiration = get_options_chain(ticker)

    if stock_data is not None:
        current_price = stock_data['Close'].iloc[-1]
        st.sidebar.metric("Current Price", f"${current_price:.2f}")

        # Update strike price based on current price if not manually set
        if strike_price == 150.0:  # Default value
            strike_price = round(current_price, 1)

        # Calculate time to expiration
        if days_to_expiry == 30:  # Use real expiration if available
            if expiration:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_expiry = (exp_date - datetime.now()).days
                days_to_expiry = max(1, min(days_to_expiry, 365))  # Ensure valid range

        time_to_expiry = days_to_expiry / 365.0

        # Calculate option metrics
        metrics = calculate_option_metrics(
            current_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, option_type.lower()
        )

        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"{option_type} Option Pricing Results")

            # Price comparison table
            price_data = {
                'Model': ['Black-Scholes', 'Binomial (European)', 'Binomial (American)'],
                'Price': [
                    metrics['black_scholes']['price'],
                    metrics['binomial_european'],
                    metrics['binomial_american']
                ]
            }

            price_df = pd.DataFrame(price_data)
            st.dataframe(price_df, use_container_width=True)

            # Greeks display
            st.subheader("Greeks Analysis")
            greeks = metrics['black_scholes']['greeks']

            greek_cols = st.columns(3)
            with greek_cols[0]:
                if option_type == "Call":
                    st.metric("Delta", f"{greeks['delta_call']:.4f}")
                else:
                    st.metric("Delta", f"{greeks['delta_put']:.4f}")
                st.metric("Gamma", f"{greeks['gamma']:.4f}")

            with greek_cols[1]:
                st.metric("Vega", f"{greeks['vega']:.4f}")
                if option_type == "Call":
                    st.metric("Theta", f"{greeks['theta_call']:.4f}")
                else:
                    st.metric("Theta", f"{greeks['theta_put']:.4f}")

            with greek_cols[2]:
                if option_type == "Call":
                    st.metric("Rho", f"{greeks['rho_call']:.4f}")
                else:
                    st.metric("Rho", f"{greeks['rho_put']:.4f}")
                st.metric("Implied Volatility", f"{volatility*100:.1f}%")

        with col2:
            st.subheader("Market Data")

            if stock_info:
                # Display key metrics
                if 'regularMarketPreviousClose' in stock_info:
                    prev_close = stock_info['regularMarketPreviousClose']
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    st.metric(
                        "Daily Change",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )

            # Display options chain summary if available
            if options_calls is not None and not options_calls.empty:
                st.subheader("Options Chain Summary")

                # Find at-the-money options
                atm_calls = options_calls[
                    (options_calls['strike'] >= current_price * 0.95) &
                    (options_calls['strike'] <= current_price * 1.05)
                ]

                if not atm_calls.empty:
                    st.write("**Near ATM Calls:**")
                    for _, row in atm_calls.head(3).iterrows():
                        st.write(f"Strike ${row['strike']:.2f}: ${row['lastPrice']:.2f}")

        # Charts section
        st.subheader("Visualizations")

        # Create strike range for charts
        strike_range = np.linspace(current_price * 0.7, current_price * 1.3, 50)

        # Calculate prices for all strikes
        bsm = BlackScholesMerton()
        call_prices = [
            bsm.call_price(current_price, K, time_to_expiry, risk_free_rate, volatility)
            for K in strike_range
        ]
        put_prices = [
            bsm.put_price(current_price, K, time_to_expiry, risk_free_rate, volatility)
            for K in strike_range
        ]

        # Price chart
        price_chart = create_price_chart(
            current_price, strike_range, call_prices, put_prices, strike_price
        )
        st.plotly_chart(price_chart, use_container_width=True)

        # Greeks heatmap
        st.subheader("Greeks Heatmap")
        greeks_chart = create_greeks_heatmap(greeks)
        st.plotly_chart(greeks_chart, use_container_width=True)

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")

        sensitivity_col1, sensitivity_col2 = st.columns(2)

        with sensitivity_col1:
            # Volatility sensitivity
            vol_range = np.linspace(0.1, 0.6, 20)
            call_prices_vol = [
                bsm.call_price(current_price, strike_price, time_to_expiry, risk_free_rate, vol)
                for vol in vol_range
            ]

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_range * 100,
                y=call_prices_vol,
                mode='lines',
                name='Call Price',
                line=dict(color='blue', width=2)
            ))
            fig_vol.update_layout(
                title="Option Price vs Volatility",
                xaxis_title="Volatility (%)",
                yaxis_title="Option Price",
                height=300
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with sensitivity_col2:
            # Time sensitivity
            time_range = np.linspace(0.01, time_to_expiry, 20)
            call_prices_time = [
                bsm.call_price(current_price, strike_price, T, risk_free_rate, volatility)
                for T in time_range
            ]

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=time_range * 365,
                y=call_prices_time,
                mode='lines',
                name='Call Price',
                line=dict(color='red', width=2)
            ))
            fig_time.update_layout(
                title="Option Price vs Time to Expiry",
                xaxis_title="Days to Expiry",
                yaxis_title="Option Price",
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)

    else:
        st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()