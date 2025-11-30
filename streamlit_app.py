import streamlit as st
from playwright.sync_api import sync_playwright
import csv
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime
import warnings
import subprocess
import sys

# Auto-install Playwright browsers on first run
@st.cache_resource
def install_playwright():
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
        # Ignore deps install failure - not critical
        subprocess.run([sys.executable, "-m", "playwright", "install-deps", "chromium"], 
                      capture_output=True)  # Remove check=True
    except:
        pass  # Silently fail

install_playwright()

warnings.filterwarnings('ignore', message='.*use_container_width.*')

# Page configuration
st.set_page_config(
    page_title="Stock Scanner Dashboard",
    page_icon="📈",
    layout="wide"
)

def scrape_symbols():
    """Scrape symbols from chartink using Playwright"""
    url = "https://chartink.com/screener/within-2-of-52"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            permissions=["clipboard-read", "clipboard-write"]
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle")
            
            # Click Copy button
            page.click("//div[contains(@class,'secondary-button') and .//span[normalize-space()='Copy']]")
            
            # Click symbols option
            page.click("//span[span[normalize-space()='symbols']]")
            page.wait_for_timeout(1000)
            
            # Get clipboard content
            clipboard_text = page.evaluate("() => navigator.clipboard.readText()")
            symbols = [s.strip() for s in clipboard_text.split(",") if s.strip()]
            
            return len(symbols), symbols
            
        finally:
            browser.close()

def check_strategy(symbol, lookback_days=80, signal_days=5, volume_filter=False, breakout_strength_filter=False):       
    try:
        ticker = f"{symbol}.NS"
       
        warnings.filterwarnings(
            "ignore",
            message=r".*YF.download\(\) has changed argument auto_adjust.*",
        )

        df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=False)
        
        if df.empty or len(df) < 200:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Calculate 200 EMA
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # Calculate All-Time High (ATH)
        df['ATH'] = df['High'].cummax()
        df['Prior_ATH'] = df['ATH'].shift(1)

        # Track when Price was Below 200 EMA
        df['Below_200'] = df['Close'] < df['EMA_200']
        
        df['idx'] = np.arange(len(df))
        df['last_below_idx'] = df['idx'].where(df['Below_200']).ffill()
        df['days_since_below'] = df['idx'] - df['last_below_idx']
        
        # Calculate average volume (20-day)
        df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
        
        # Define Buy Signal - Base condition
        buy_signal_base = (
            (df['Close'] > df['Prior_ATH']) &
            (df['days_since_below'] <= lookback_days) &
            (df['days_since_below'] >= 0)
        )
        
        # Add Volume Filter if enabled
        if volume_filter:
            volume_condition = df['Volume'] > (df['Avg_Volume_20'] * 1.5)
            buy_signal_base = buy_signal_base & volume_condition
        
        # Add Breakout Strength Filter if enabled (close at least 1% above prior ATH)
        if breakout_strength_filter:
            breakout_strength = ((df['Close'] - df['Prior_ATH']) / df['Prior_ATH'] * 100) >= 1.0
            buy_signal_base = buy_signal_base & breakout_strength
        
        df['Buy_Signal'] = buy_signal_base

        # Check last N days (dynamic based on signal_days parameter)
        recent_data = df.tail(signal_days)                   
        
        if recent_data['Buy_Signal'].any():
            signal_rows = recent_data[recent_data['Buy_Signal']]
            current_close = df['Close'].iloc[-1]
            signals = []
            for idx, row in signal_rows.iterrows():
                signal_idx = int(row['idx'])
                buy_price = float(row['Close'])
                ema_200 = float(row['EMA_200']) if not np.isnan(row['EMA_200']) else None
                prior_ath = float(row['Prior_ATH']) if not np.isnan(row['Prior_ATH']) else None
                days_since_signal = int(df['idx'].iloc[-1] - signal_idx)
                diff = current_close - buy_price
                diff_pct = (diff / buy_price * 100) if buy_price != 0 else None

                signals.append({
                    'signal_date': pd.to_datetime(idx).strftime('%Y-%m-%d'),
                    'buy_price': round(buy_price, 4),
                    'current_price': round(float(current_close), 4),
                    'diff': round(diff, 4),
                    'diff_pct': round(diff_pct, 2) if diff_pct is not None else None,
                    'days_since_signal': days_since_signal,
                    'ema_200_at_signal': round(ema_200, 4) if ema_200 is not None else None,
                    'prior_ath_at_signal': round(prior_ath, 4) if prior_ath is not None else None,
                    'days_since_below_at_signal': int(row['days_since_below']) if not np.isnan(row['days_since_below']) else None,
                })

            signals_sorted = sorted(signals, key=lambda s: s['signal_date'])
            return [signals_sorted[0]]
            
        return None

    except Exception as e:
        return None

def scan_stocks(symbols, progress_bar, status_text, signal_days=5, lookback_days=80, volume_filter=False, breakout_strength_filter=False):
    """Scan stocks for buy signals"""
    execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)

        signals = check_strategy(symbol, lookback_days=lookback_days, signal_days=signal_days, 
                                volume_filter=volume_filter, breakout_strength_filter=breakout_strength_filter)

        if signals:
            for s in signals:
                results.append({
                    'Execution_Time': execution_time,
                    'Symbol': symbol,
                    'Signal Date': s['signal_date'],
                    'Buy Price': round(s['buy_price'], 2),
                    'Current Price': round(s['current_price'], 2),
                    'Diff': round(s['diff'], 2),
                    'Diff %': round(s['diff_pct'], 2) if s['diff_pct'] is not None else None,
                    'Days Since Signal': s['days_since_signal'],
                    'EMA200 at Signal': round(s['ema_200_at_signal'], 2) if s['ema_200_at_signal'] is not None else None,
                    'Prior ATH at Signal': round(s['prior_ath_at_signal'], 2) if s['prior_ath_at_signal'] is not None else None,
                    'Days Since Below at Signal': s['days_since_below_at_signal'],
                })

    if results:
        result_df = pd.DataFrame(results)
        return result_df
    else:
        return None


# Streamlit UI
st.title("📈 Stock Scanner Dashboard by Darshan")
st.markdown("### ATH Breakout Strategy with 200 EMA")

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This scanner finds stocks that:\n- Broke all-time high\n- Were below 200 EMA within configurable days")
    
    if st.button("ℹ️ About Strategy"):
        st.write("""
        **Strategy Logic:**
        1. Stock breaks its all-time high (ATH)
        2. Stock was below 200 EMA within last N days (configurable: 40-120)
        3. Signal generated in last M trading days (configurable: 1-30)
        """)

# Main content
# Input fields side by side
col_input1, col_input2 = st.columns(2)

with col_input1:
    signal_days = st.number_input(
        "📅 Look back for signals (days)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Number of past trading days to check for buy signals (max: 30)"
    )
    st.caption(f"Scanner will check signals from last **{signal_days}** trading days")

with col_input2:
    lookback_days = st.number_input(
        "📊 Days since below 200 EMA",
        min_value=40,
        max_value=120,
        value=80,
        step=5,
        help="Maximum days since price was below 200 EMA (range: 40-120)"
    )
    st.caption(f"Stocks must have been below 200 EMA within last **{lookback_days}** days")

# Filter checkboxes
st.markdown("### 🎯 Advanced Filters")
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    volume_filter = st.checkbox(
        "📊 Volume Confirmation",
        value=False,
        help="Require volume to be 1.5x above 20-day average on breakout day"
    )
    if volume_filter:
        st.caption("✅ Volume must be >1.5x average")

with col_filter2:
    breakout_strength_filter = st.checkbox(
        "💪 Breakout Strength",
        value=False,
        help="Require close to be at least 1% above prior ATH"
    )
    if breakout_strength_filter:
        st.caption("✅ Close must be ≥1% above ATH")

st.divider()

# Fetch or use cached symbols
if 'symbols' not in st.session_state:
    with st.spinner("Fetching symbols from ChartInk for this session..."):
        try:
            num_symbols, symbols = scrape_symbols()
            st.session_state['symbols'] = symbols
            st.session_state['num_symbols'] = num_symbols
            st.success(f"✅ Loaded {num_symbols} symbols for this session")
        except Exception as e:
            st.error(f"❌ Failed to fetch symbols: {str(e)}")
            st.stop()
else:
    st.info(f"ℹ️ Using {st.session_state['num_symbols']} symbols from session cache")
    with st.expander("View symbols"):
        st.write(st.session_state['symbols'])

if st.button("🚀 Run Full Scanner", type="primary", use_container_width=True):
    with st.spinner("Starting scanner..."):
        try:
            # Use symbols from session state
            symbols = st.session_state['symbols']
            
            # Step 2: Scan stocks
            filters_text = []
            if volume_filter:
                filters_text.append("Volume Filter ON")
            if breakout_strength_filter:
                filters_text.append("Breakout Strength ON")
            filters_display = f" ({', '.join(filters_text)})" if filters_text else ""
            
            st.info(f"**Analyzing stocks** for buy signals (last {signal_days} days, below 200 EMA within {lookback_days} days){filters_display}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result_df = scan_stocks(symbols, progress_bar, status_text, signal_days=signal_days, 
                                  lookback_days=lookback_days, volume_filter=volume_filter, 
                                  breakout_strength_filter=breakout_strength_filter)
            
            progress_bar.empty()
            status_text.empty()
            
            if result_df is not None and not result_df.empty:
                # Store results in session state
                st.session_state['results'] = result_df
                st.session_state['signal_days_used'] = signal_days
                st.session_state['lookback_days_used'] = lookback_days
                st.session_state['volume_filter_used'] = volume_filter
                st.session_state['breakout_filter_used'] = breakout_strength_filter
                
                st.success(f"🎉 Found {len(result_df)} buy signals!")
                
                # Display metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Signals", len(result_df))
                with col_b:
                    avg_gain = result_df['Diff %'].mean()
                    st.metric("Avg Gain %", f"{avg_gain:.2f}%")
                with col_c:
                    positive_signals = len(result_df[result_df['Diff %'] > 0])
                    st.metric("Positive Signals", positive_signals)
                
                st.divider()
                
                # Display table
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Diff %": st.column_config.NumberColumn(
                            "Diff %",
                            format="%.2f%%",
                        ),
                    }
                )
                
                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"buy_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning(f"No buy signals found in the last {signal_days} days.")
                
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")

st.divider()

# View Previous Results button - now below the main results
if st.button("📊 View Previous Results", use_container_width=True):
    if 'results' in st.session_state and st.session_state['results'] is not None:
        df = st.session_state['results']
        days_used = st.session_state.get('signal_days_used', 'N/A')
        lookback_used = st.session_state.get('lookback_days_used', 'N/A')
        vol_filter = st.session_state.get('volume_filter_used', False)
        breakout_filter = st.session_state.get('breakout_filter_used', False)
        
        filters_used = []
        if vol_filter:
            filters_used.append("Volume Filter")
        if breakout_filter:
            filters_used.append("Breakout Strength")
        filters_text = f" | Filters: {', '.join(filters_used)}" if filters_used else " | No filters"
        
        st.success(f"Loaded {len(df)} signals from previous run (signal days: {days_used}, lookback days: {lookback_used}{filters_text})")
        
        # Display metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Signals", len(df))
        with col_b:
            avg_gain = df['Diff %'].mean()
            st.metric("Avg Gain %", f"{avg_gain:.2f}%")
        with col_c:
            positive_signals = len(df[df['Diff %'] > 0])
            st.metric("Positive Signals", positive_signals)
        
        st.divider()
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"previous_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No previous results found. Run the scanner first!")

st.divider()

# Clear button at bottom
if st.button("🗑️ Clear Results", use_container_width=False):
    if 'results' in st.session_state:
        del st.session_state['results']
        if 'signal_days_used' in st.session_state:
            del st.session_state['signal_days_used']
        if 'lookback_days_used' in st.session_state:
            del st.session_state['lookback_days_used']
        if 'volume_filter_used' in st.session_state:
            del st.session_state['volume_filter_used']
        if 'breakout_filter_used' in st.session_state:
            del st.session_state['breakout_filter_used']
        st.success("✅ Results cleared!")
        st.rerun()
    else:
        st.info("No results to clear.")

# Add button to refresh symbols
if st.button("🔄 Refresh Symbols from ChartInk", use_container_width=False):
    with st.spinner("Fetching fresh symbols from ChartInk..."):
        try:
            num_symbols, symbols = scrape_symbols()
            st.session_state['symbols'] = symbols
            st.session_state['num_symbols'] = num_symbols
            st.success(f"✅ Refreshed! Loaded {num_symbols} new symbols")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to refresh symbols: {str(e)}")

# Footer
st.divider()
st.caption("Stock Scanner made by Darshan Ramani with ❤️ | Data source: ChartInk & Yahoo Finance")