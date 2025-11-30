# 📈 BreakoutScout

A powerful stock scanner that identifies All-Time High (ATH) breakout opportunities using the 200 EMA strategy. Built with Streamlit for a beautiful, interactive dashboard experience.

##  [Live Demo](https://breakout-scout.streamlit.app/)

## 🎯 Strategy

BreakoutScout finds stocks that meet these criteria:
- **Broke All-Time High (ATH)** - Stock price exceeds its previous all-time high
- **Recent Below 200 EMA** - Was below the 200-day EMA within the last 80 days
- **Fresh Signals** - Generated within the last 5 trading days

This strategy identifies momentum stocks emerging from consolidation phases.

## ✨ Features

- 🔍 **Automated Scanning** - Scrapes stocks from ChartInk and analyzes them automatically
- 📊 **Beautiful Dashboard** - Interactive Streamlit UI with real-time progress tracking
- 📈 **Performance Metrics** - View total signals, average gains, and positive signals
- 💾 **Session Storage** - Results stored in memory (no local files)
- 📥 **Easy Export** - Download results as CSV with timestamp
- 🎨 **Responsive Design** - Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Chrome/Chromium browser (for local development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DarshanR1510/breakout-scout.git
   cd breakout-scout
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (if using Playwright)
   ```bash
   playwright install chromium
   ```

4. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open browser**
   - App will automatically open at `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit
playwright
pandas
yfinance
numpy
```

## 🎮 Usage

1. **Click "🚀 Run Full Scanner"**
   - Automatically scrapes symbols from ChartInk
   - Analyzes each stock for buy signals
   - Displays results in an interactive table

2. **View Results**
   - See metrics: Total signals, Average gain %, Positive signals
   - Browse detailed table with all signal information
   - Download results as CSV

3. **View Previous Results**
   - Click "📊 View Previous Results" to see last scan
   - Download previous results

4. **Clear Results**
   - Click "🗑️ Clear Results" to reset session data

## 📊 Output Columns

| Column | Description |
|--------|-------------|
| Execution_Time | When the scan was performed |
| Symbol | Stock ticker symbol |
| Signal_Date | Date when buy signal was generated |
| Buy_Price | Price at signal generation |
| Current_Price | Latest closing price |
| Diff | Price difference (Current - Buy) |
| Diff % | Percentage gain/loss |
| Days Since Signal | Trading days since signal |
| EMA200 at Signal | 200 EMA value at signal time |
| Prior ATH at Signal | Previous ATH before breakout |
| Days Since Below at Signal | Days since stock was below 200 EMA |

## 🌐 Deployment

### Deploy on Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Create `packages.txt`** (for system dependencies)
   ```txt
   chromium
   chromium-driver
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `streamlit_app.py` as main file
   - Click "Deploy"

### Alternative Deployment Options

- **Render.com** - Great Selenium/Playwright support
- **Railway.app** - Easy deployment with auto-scaling
- **Heroku** - Requires buildpacks for Chrome
- **PythonAnywhere** - Good for scheduled runs

## 🛠️ Configuration

### Adjust Lookback Period

In `streamlit_app.py`, modify the `lookback_days` parameter:

```python
def check_strategy(symbol, lookback_days=80):  # Change 80 to your value
```

### Change Data Source

Update the URL in `scrape_symbols()`:

```python
url = "https://chartink.com/screener/your-screener-name"
```

## 📝 Project Structure

```
breakout-scout/
├── streamlit_app.py      # Main application file
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies (for deployment)
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Always do your own research and consult with a financial advisor before making investment decisions.

## 🙏 Acknowledgments

- **ChartInk** - Stock screening data source
- **Yahoo Finance** - Historical stock data via yfinance
- **Streamlit** - Web app framework

## 📧 Contact

- GitHub: [@DarshanR1510](https://github.com/DarshanR1510)
- Project Link: [https://github.com/DarshanR1510/breakout-scout](https://github.com/DarshanR1510/breakout-scout)

---

Made with ❤️ by Darshan | Star ⭐ this repo if you find it helpful!