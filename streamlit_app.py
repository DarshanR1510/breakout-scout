import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', message='.*use_container_width.*')

# Page configuration
st.set_page_config(
    page_title="Stock Scanner Dashboard",
    page_icon="📈",
    layout="wide"
)

def scrape_symbols():
    """Scrape symbols from chartink and return list"""
    url = "https://chartink.com/screener/within-2-of-52"

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    driver.execute_cdp_cmd('Browser.grantPermissions', {
        'origin': 'https://chartink.com',
        'permissions': ['clipboardReadWrite', 'clipboardSanitizedWrite']
    })

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)

        # Click "Copy" button
        copy_button = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//div[contains(@class,'secondary-button') and .//span[normalize-space()='Copy']]"
                )
            )
        )
        copy_button.click()

        # Click "symbols" in the popup
        symbols_option = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//span[span[normalize-space()='symbols']]"
                )
            )
        )
        symbols_option.click()
        time.sleep(1)

        # Read from clipboard
        clipboard_text = driver.execute_script("""
            return navigator.clipboard.readText();
        """)
        
        symbols = [s.strip() for s in clipboard_text.split(",") if s.strip()]

        return len(symbols), symbols

    finally:
        driver.quit()

def check_strategy(symbol, lookback_days=80):       
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

        df = df[['Open', 'High', 'Low', 'Close']].copy()

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
        
        # Define Buy Signal
        df['Buy_Signal'] = (
            (df['Close'] > df['Prior_ATH']) &
            (df['days_since_below'] <= lookback_days) &
            (df['days_since_below'] >= 0)
        )

        recent_data = df.tail(5)                   
        
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

def scan_stocks(symbols, progress_bar, status_text):
    """Scan stocks for buy signals"""
    execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)

        signals = check_strategy(symbol)

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
    st.info("This scanner finds stocks that:\n- Broke all-time high\n- Were below 200 EMA within last 80 days")
    
    if st.button("ℹ️ About Strategy"):
        st.write("""
        **Strategy Logic:**
        1. Stock breaks its all-time high (ATH)
        2. Stock was below 200 EMA within last 80 days
        3. Signal generated in last 5 trading days
        """)

# Main content
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Run Full Scanner", type="primary", use_container_width=True):
        with st.spinner("Starting scanner..."):
            try:
                # Step 1: Scrape symbols
                st.info("**Step 1:** Scraping symbols from ChartInk...")
                num_symbols, symbols = scrape_symbols()
                st.success(f"✅ Found {num_symbols} symbols")
                
                # Store in session state
                st.session_state['symbols'] = symbols
                
                # Show symbols in expander
                with st.expander("View scraped symbols"):
                    st.write(symbols)
                
                # Step 2: Scan stocks
                st.info("**Step 2:** Analyzing stocks for buy signals...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result_df = scan_stocks(symbols, progress_bar, status_text)
                
                progress_bar.empty()
                status_text.empty()
                
                if result_df is not None and not result_df.empty:
                    # Store results in session state
                    st.session_state['results'] = result_df
                    
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
                    st.warning("No buy signals found in the last 5 days.")
                    
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")

with col2:
    if st.button("📊 View Previous Results", use_container_width=True):
        if 'results' in st.session_state and st.session_state['results'] is not None:
            df = st.session_state['results']
            st.success(f"Loaded {len(df)} signals from previous run")
            
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
        if 'symbols' in st.session_state:
            del st.session_state['symbols']
        st.success("✅ Results cleared!")
        st.rerun()
    else:
        st.info("No results to clear.")

# Footer
st.divider()
st.caption("Stock Scanner v1.0 | Data source: ChartInk & Yahoo Finance")