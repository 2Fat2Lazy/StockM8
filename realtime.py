import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time as dt_time
import time
import pytz

st.set_page_config(page_title="📈 Real-Time Stock Tracker", layout="wide")

# --- MARKET HOURS CHECK ---
def is_market_open():
    """Check if US stock market is currently open"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if between 9:30 AM and 4:00 PM ET
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    current_time = now.time()
    
    return market_open <= current_time <= market_close

# --- AUTO-REFRESH CONFIGURATION ---
st.sidebar.header("⚙️ Settings")

# Smart auto-refresh
auto_refresh_mode = st.sidebar.selectbox(
    "🔄 Auto-Refresh Mode",
    ["Off", "Manual (Custom Interval)", "Smart (Market Hours Only)", "Aggressive (Always On)"]
)

refresh_interval = 30  # default

if auto_refresh_mode == "Manual (Custom Interval)":
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, 30)
    st.sidebar.info(f"Refreshing every {refresh_interval} seconds")
    time.sleep(refresh_interval)
    st.rerun()

elif auto_refresh_mode == "Smart (Market Hours Only)":
    if is_market_open():
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 120, 15)
        st.sidebar.success(f"🟢 Market OPEN - Refreshing every {refresh_interval}s")
        time.sleep(refresh_interval)
        st.rerun()
    else:
        st.sidebar.warning("🔴 Market CLOSED - Auto-refresh paused")
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        st.sidebar.info(f"Current ET time: {now.strftime('%H:%M:%S')}")
        st.sidebar.info("Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri)")

elif auto_refresh_mode == "Aggressive (Always On)":
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    st.sidebar.warning(f"⚠️ Refreshing every {refresh_interval}s (watch API limits!)")
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.markdown("---")

# Ticker selection
ticker_input = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
tickers = [t.strip() for t in ticker_input.split(",")]

period = st.sidebar.selectbox(
    "Data Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=2
)

# For intraday data
interval = "1m" if period in ["1d", "5d"] else "1d"

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Technical Indicators")

selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    ["SMA_20", "SMA_50", "EMA_20", "Bollinger Bands", "RSI", "MACD", "Volume"],
    default=["SMA_20", "RSI", "Volume"]
)

# --- FETCH DATA ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.copy()
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

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
    
    return df

# --- MAIN APP ---
st.title("📈 Real-Time Stock Tracker")

# Market status banner
if is_market_open():
    st.success("🟢 **US Market is OPEN**")
else:
    st.info("🔴 **US Market is CLOSED** - Showing last available data")

ticker = tickers[0]

# Fetch data
data = get_data(ticker, period, interval)

if data.empty:
    st.error(f"❌ No data returned for ticker '{ticker}'.")
    st.stop()

if "Close" not in data.columns:
    st.error(f"❌ No 'Close' column found.")
    st.stop()

# Calculate indicators
data = calculate_indicators(data, selected_indicators)

# Get company info
info = get_ticker_info(ticker)

# --- METRICS ---
col1, col2, col3, col4, col5 = st.columns(5)

current_price = data['Close'].iloc[-1]
previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
price_change = current_price - previous_price
price_change_pct = (price_change / previous_price) * 100

col1.metric(
    "Current Price", 
    f"${current_price:.2f}",
    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
)

if len(data) > 0:
    day_high = data['High'].max()
    day_low = data['Low'].min()
    col2.metric("Day High", f"${day_high:.2f}")
    col3.metric("Day Low", f"${day_low:.2f}")

if "Volume" in data.columns:
    total_volume = data['Volume'].sum()
    col4.metric("Total Volume", f"{total_volume/1e6:.1f}M")

if info:
    col5.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")

# Last update time
st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Next refresh in {refresh_interval}s" if auto_refresh_mode != "Off" else f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- CHART ---
st.subheader(f"📊 {ticker} - {period.upper()} Chart")

fig = make_subplots(
    rows=3 if "RSI" in selected_indicators else 2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2] if "RSI" in selected_indicators else [0.7, 0.3],
    subplot_titles=(f"{ticker} Price", "Volume", "RSI") if "RSI" in selected_indicators else (f"{ticker} Price", "Volume")
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="OHLC",
    increasing_line_color='green',
    decreasing_line_color='red'
), row=1, col=1)

# Add indicators to price chart
if "SMA_20" in data.columns:
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data["SMA_20"], 
        name="SMA 20", 
        line=dict(color='orange', width=2)
    ), row=1, col=1)

if "EMA_20" in data.columns:
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data["EMA_20"], 
        name="EMA 20", 
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)

if "BB_Upper" in data.columns:
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data["BB_Upper"], 
        name="BB Upper", 
        line=dict(color='gray', width=1, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data["BB_Lower"], 
        name="BB Lower", 
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)'
    ), row=1, col=1)

# Volume
if "Volume" in data.columns:
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name="Volume",
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)

# RSI
if "RSI" in data.columns and "RSI" in selected_indicators:
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["RSI"],
        name="RSI",
        line=dict(color='purple', width=2)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.update_layout(
    height=800,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Date/Time", row=3 if "RSI" in selected_indicators else 2, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
if "RSI" in selected_indicators:
    fig.update_yaxes(title_text="RSI", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- CURRENT SIGNALS ---
st.subheader("🎯 Live Signals")

col1, col2, col3 = st.columns(3)

with col1:
    if "RSI" in data.columns:
        current_rsi = data["RSI"].iloc[-1]
        if pd.notna(current_rsi):
            if current_rsi > 70:
                st.error(f"🔴 **RSI: {current_rsi:.2f}**\nOVERBOUGHT")
            elif current_rsi < 30:
                st.success(f"🟢 **RSI: {current_rsi:.2f}**\nOVERSOLD")
            else:
                st.info(f"⚪ **RSI: {current_rsi:.2f}**\nNEUTRAL")

with col2:
    if "SMA_20" in data.columns:
        sma_20 = data["SMA_20"].iloc[-1]
        if pd.notna(sma_20):
            if current_price > sma_20:
                st.success(f"🟢 **Price > SMA(20)**\nBULLISH\n${current_price:.2f} > ${sma_20:.2f}")
            else:
                st.error(f"🔴 **Price < SMA(20)**\nBEARISH\n${current_price:.2f} < ${sma_20:.2f}")

with col3:
    # Price momentum
    if len(data) >= 5:
        price_5_ago = data['Close'].iloc[-5]
        momentum = ((current_price - price_5_ago) / price_5_ago) * 100
        if momentum > 0:
            st.success(f"🟢 **5-Period Momentum**\n+{momentum:.2f}%")
        else:
            st.error(f"🔴 **5-Period Momentum**\n{momentum:.2f}%")

# --- RECENT DATA ---
with st.expander("📊 Recent Data (Last 20 Bars)"):
    display_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "RSI" in data.columns:
        display_cols.append("RSI")
    if "SMA_20" in data.columns:
        display_cols.append("SMA_20")
    
    st.dataframe(data[display_cols].tail(20), use_container_width=True)

# Manual refresh button
st.markdown("---")
if st.button("🔄 Refresh Now"):
    st.rerun()