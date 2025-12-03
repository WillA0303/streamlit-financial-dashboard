# Macro–Equity Dashboard

Interactive macro and equity market dashboard built in Python and Streamlit.

This app lets you explore key macro indicators alongside equity index performance, to see how inflation, unemployment and interest rates interact with stock prices over time.

## Features

- Macro data:
  - CPI, unemployment and policy rate for selected economies
  - Long term historical charts
- Equity markets:
  - Custom list of tickers via Yahoo Finance
  - Price charts, returns and drawdowns
- Macro vs equity:
  - Overlays of equity indices with macro series
  - Basic return statistics
- Export and reproducibility:
  - Simple layout so users can adapt it for their own analysis

## Tech stack

- Python
- Streamlit
- pandas
- yfinance
- requests
- matplotlib / plotly (adjust if needed)

## How to run locally

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/macro-equity-dashboard.git
   cd macro-equity-dashboard
