import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

st.set_page_config(page_title="📈 Advanced Stock Analysis", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Settings")

# Ticker selection
ticker_input = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
tickers = [t.strip() for t in ticker_input.split(",")]

period = st.sidebar.selectbox(
    "Data Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    index=3
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Technical Indicators")

# Indicator selection
selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    ["SMA_20", "SMA_50", "EMA_20", "Bollinger Bands", "RSI", "MACD", "Stochastic", "ATR", "OBV"],
    default=["SMA_20", "RSI", "MACD"]
)

# Strategy parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Settings")
short_sma = st.sidebar.slider("Short SMA Window", 5, 50, 20)
long_sma = st.sidebar.slider("Long SMA Window", 20, 200, 50)
rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30)

# --- FETCH DATA ---
@st.cache_data
def get_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.copy()
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

@st.cache_data
def get_news(ticker):
    try:
        return yf.Ticker(ticker).news[:5]
    except:
        return []

# --- CALCULATE INDICATORS ---
def calculate_indicators(data, indicators):
    df = data.copy()
    close = df["Close"]
    
    if "SMA_20" in indicators:
        df["SMA_20"] = ta.sma(close, length=20)
    
    if "SMA_50" in indicators:
        df["SMA_50"] = ta.sma(close, length=50)
    
    if "EMA_20" in indicators:
        df["EMA_20"] = ta.ema(close, length=20)
    
    if "Bollinger Bands" in indicators:
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None:
            df["BB_Upper"] = bbands["BBU_20_2.0"]
            df["BB_Middle"] = bbands["BBM_20_2.0"]
            df["BB_Lower"] = bbands["BBL_20_2.0"]
    
    if "RSI" in indicators:
        df["RSI"] = ta.rsi(close, length=14)
    
    if "MACD" in indicators:
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_Signal"] = macd["MACDs_12_26_9"]
            df["MACD_Hist"] = macd["MACDh_12_26_9"]
    
    if "Stochastic" in indicators:
        stoch = ta.stoch(df["High"], df["Low"], close, k=14, d=3)
        if stoch is not None:
            df["Stoch_K"] = stoch["STOCHk_14_3_3"]
            df["Stoch_D"] = stoch["STOCHd_14_3_3"]
    
    if "ATR" in indicators:
        df["ATR"] = ta.atr(df["High"], df["Low"], close, length=14)
    
    if "OBV" in indicators:
        df["OBV"] = ta.obv(close, df["Volume"])
    
    return df

# --- BACKTEST STRATEGY ---
def backtest_strategy(data, short_window, long_window):
    df = data.copy()
    df["SMA_Short"] = ta.sma(df["Close"], length=short_window)
    df["SMA_Long"] = ta.sma(df["Close"], length=long_window)
    df = df.dropna()
    
    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df["Position"] = df["Signal"].diff()
    
    # Calculate returns
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]
    df["Cumulative_Returns"] = (1 + df["Returns"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
    
    return df

# --- PERFORMANCE METRICS ---
def calculate_performance(data):
    returns = data["Close"].pct_change().dropna()
    
    total_return = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    max_price = data["Close"].max()
    min_price = data["Close"].min()
    max_date = data["Close"].idxmax()
    min_date = data["Close"].idxmin()
    
    return {
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_price": max_price,
        "min_price": min_price,
        "max_date": max_date,
        "min_date": min_date
    }

# --- MAIN APP ---
st.title("📈 Advanced Stock Analysis Dashboard")

# Handle multiple tickers
if len(tickers) > 1:
    st.info(f"📊 Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
    
    # Portfolio comparison view
    portfolio_data = {}
    for ticker in tickers:
        data = get_data(ticker, period)
        if not data.empty:
            portfolio_data[ticker] = data
    
    if portfolio_data:
        # Normalize prices to 100 for comparison
        st.subheader("📊 Portfolio Comparison (Normalized)")
        fig_portfolio = go.Figure()
        for ticker, data in portfolio_data.items():
            normalized = (data["Close"] / data["Close"].iloc[0]) * 100
            fig_portfolio.add_trace(go.Scatter(x=data.index, y=normalized, name=ticker, mode='lines'))
        
        fig_portfolio.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Normalized Price (Start = 100)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Correlation Matrix")
        close_prices = pd.DataFrame({ticker: data["Close"] for ticker, data in portfolio_data.items()})
        returns = close_prices.pct_change().dropna()
        corr_matrix = returns.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    ticker = tickers[0]  # Use first ticker for detailed analysis
    st.markdown("---")
    st.subheader(f"📊 Detailed Analysis: {ticker}")
else:
    ticker = tickers[0]

# Fetch data for main analysis
data = get_data(ticker, period)

if data.empty:
    st.error(f"❌ No data returned for ticker '{ticker}'.")
    st.stop()

if "Close" not in data.columns:
    st.error(f"❌ No 'Close' column found. Available: {', '.join(data.columns)}")
    st.stop()

# Calculate indicators
data = calculate_indicators(data, selected_indicators)

# Get company info
info = get_ticker_info(ticker)

# Create tabs for organized display
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price & Indicators", "🎯 Backtest", "📊 Fundamentals", "📰 News", "🔍 Advanced"])

# --- TAB 1: PRICE & INDICATORS ---
with tab1:
    # Performance metrics
    perf = calculate_performance(data)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    col2.metric("Total Return", f"{perf['total_return']:.2f}%")
    col3.metric("Volatility", f"{perf['volatility']:.2f}%")
    col4.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}")
    col5.metric("Max Price", f"${perf['max_price']:.2f}")
    
    # Candlestick chart with indicators
    st.subheader("📊 Price Chart")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & Volume", "RSI", "MACD")
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="OHLC"
    ), row=1, col=1)
    
    # Add selected indicators
    if "SMA_20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color='orange')), row=1, col=1)
    
    if "SMA_50" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color='purple')), row=1, col=1)
    
    if "EMA_20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"], name="EMA 20", line=dict(color='green', dash='dash')), row=1, col=1)
    
    if "BB_Upper" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper", line=dict(color='gray', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower", line=dict(color='gray', dash='dot'), fill='tonexty'), row=1, col=1)
    
    # RSI
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if "MACD" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram", marker_color='gray'), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current signals
    st.subheader("🎯 Current Signals")
    col1, col2 = st.columns(2)
    
    with col1:
        if "RSI" in data.columns:
            current_rsi = data["RSI"].iloc[-1]
            if current_rsi > rsi_overbought:
                st.error(f"🔴 RSI: {current_rsi:.2f} - OVERBOUGHT")
            elif current_rsi < rsi_oversold:
                st.success(f"🟢 RSI: {current_rsi:.2f} - OVERSOLD")
            else:
                st.info(f"⚪ RSI: {current_rsi:.2f} - NEUTRAL")
    
    with col2:
        if "SMA_20" in data.columns and "SMA_50" in data.columns:
            if data["SMA_20"].iloc[-1] > data["SMA_50"].iloc[-1]:
                st.success("🟢 SMA 20 > SMA 50 - BULLISH")
            else:
                st.error("🔴 SMA 20 < SMA 50 - BEARISH")

# --- TAB 2: BACKTEST ---
with tab2:
    st.subheader("🎯 Strategy Backtesting")
    st.write(f"Testing SMA crossover strategy: Short={short_sma}, Long={long_sma}")
    
    backtest_data = backtest_strategy(data, short_sma, long_sma)
    
    # Performance metrics
    strategy_return = (backtest_data["Cumulative_Strategy"].iloc[-1] - 1) * 100
    buy_hold_return = (backtest_data["Cumulative_Returns"].iloc[-1] - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Return", f"{strategy_return:.2f}%")
    col2.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
    col3.metric("Outperformance", f"{strategy_return - buy_hold_return:.2f}%")
    
    # Equity curve
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(
        x=backtest_data.index,
        y=backtest_data["Cumulative_Returns"],
        name="Buy & Hold",
        line=dict(color='blue')
    ))
    fig_backtest.add_trace(go.Scatter(
        x=backtest_data.index,
        y=backtest_data["Cumulative_Strategy"],
        name="Strategy",
        line=dict(color='green')
    ))
    
    # Mark buy/sell signals
    buys = backtest_data[backtest_data["Position"] == 1]
    sells = backtest_data[backtest_data["Position"] == -1]
    
    fig_backtest.add_trace(go.Scatter(
        x=buys.index,
        y=buys["Cumulative_Strategy"],
        mode='markers',
        name="Buy",
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    fig_backtest.add_trace(go.Scatter(
        x=sells.index,
        y=sells["Cumulative_Strategy"],
        mode='markers',
        name="Sell",
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    
    fig_backtest.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

# --- TAB 3: FUNDAMENTALS ---
with tab3:
    st.subheader("📊 Company Fundamentals")
    
    if info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            st.metric("EPS", f"${info.get('trailingEps', 'N/A')}")
        
        with col2:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
        
        with col3:
            st.metric("Revenue", f"${info.get('totalRevenue', 0)/1e9:.2f}B")
            st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
            st.metric("Beta", f"{info.get('beta', 'N/A')}")
        
        with st.expander("📋 Full Company Info"):
            st.write(f"**Company:** {info.get('longName', ticker)}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Description:** {info.get('longBusinessSummary', 'N/A')}")
    else:
        st.warning("Fundamental data not available")

# --- TAB 4: NEWS ---
with tab4:
    st.subheader("📰 Recent News")
    news = get_news(ticker)
    
    if news:
        for article in news:
            with st.expander(f"📄 {article.get('title', 'No Title')}"):
                st.write(f"**Publisher:** {article.get('publisher', 'Unknown')}")
                st.write(f"**Link:** {article.get('link', '#')}")
                st.write(article.get('summary', 'No summary available'))
    else:
        st.info("No recent news available")

# --- TAB 5: ADVANCED ---
with tab5:
    st.subheader("🔍 Advanced Analysis")
    
    # Volume analysis
    if "OBV" in data.columns:
        st.write("**On-Balance Volume (OBV)**")
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=data.index, y=data["OBV"], name="OBV", fill='tozeroy'))
        fig_obv.update_layout(height=300)
        st.plotly_chart(fig_obv, use_container_width=True)
    
    # Volatility analysis
    if "ATR" in data.columns:
        st.write("**Average True Range (ATR) - Volatility**")
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=data.index, y=data["ATR"], name="ATR", line=dict(color='red')))
        fig_atr.update_layout(height=300)
        st.plotly_chart(fig_atr, use_container_width=True)
    
    # Recent data table
    with st.expander("📊 View Raw Data"):
        st.dataframe(data.tail(50), use_container_width=True)