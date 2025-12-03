##############################################
# app.py – Macro–Equity Dashboard (Streamlit)
##############################################

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd
import yfinance as yf
import streamlit as st
import requests

# -----------------------
# Configuration
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_PATH = PROJECT_ROOT / "data" / "macro"

MACRO_SERIES: Dict[str, Dict[str, str]] = {
    "United States": {
        "cpi": "CPIAUCSL",        # CPI for All Urban Consumers: All Items
        "unemployment": "UNRATE", # Unemployment Rate
        "rate": "FEDFUNDS",       # Effective Federal Funds Rate
    },
    "United Kingdom": {
        "cpi": "GBRCPIALLMINMEI",     # CPI All Items
        "unemployment": "LRHUTTTTGBQ156S",  # Harmonized unemployment rate
        "rate": "BOEBASE",            # Bank of England Bank Rate
    },
    "Euro Area": {
        "cpi": "CP0000EZ19M086NEST",  # Euro Area CPI All Items
        "unemployment": "LRHUTTTTEZM156S",  # Harmonized unemployment rate
        "rate": "ECBDFR",             # ECB Deposit Facility Rate
    },
}

# Global top 10 by market cap (mapped to yfinance tickers)
PORTFOLIO_TICKERS: List[str] = [
    "NVDA",      # Nvidia
    "AAPL",      # Apple
    "GOOGL",     # Alphabet
    "MSFT",      # Microsoft
    "AMZN",      # Amazon
    "AVGO",      # Broadcom
    "META",      # Meta
    "2222.SR",   # Saudi Aramco
    "TSM",       # TSMC
    "TSLA",      # Tesla
]

# -----------------------
# Utility helpers
# -----------------------

def _to_float(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _find_row_label(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first row label in df.index that contains any of the candidate strings."""
    if df is None or df.empty:
        return None
    labels = [str(idx) for idx in df.index]
    lower_labels = [str(idx).lower() for idx in df.index]

    for cand in candidates:
        c = cand.lower()
        for lbl, lower_lbl in zip(labels, lower_labels):
            if c in lower_lbl:
                return lbl
    return None


def _find_equity_label(bs_df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Total stockholders' equity",
        "Total stockholders equity",
        "Total stockholder equity",
        "Total shareholders' equity",
        "Total shareholders equity",
        "Total shareholder equity",
        "Total equity",
    ]
    return _find_row_label(bs_df, candidates)


# -----------------------
# Macro engine
# -----------------------

def _load_from_csv(filename: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(MACRO_PATH / filename, parse_dates=["Date"])
    df = (
        df.rename(columns={"Value": column})
        .set_index("Date")
        .sort_index()
        .dropna()
    )
    return df


def fetch_macro_series(series_id: str, name: str) -> pd.DataFrame:
    """
    Fetch a single FRED series by downloading its CSV directly.

    Uses the public FRED CSV endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    df = pd.read_csv(pd.compat.StringIO(resp.text), parse_dates=["DATE"])
    df = (
        df.rename(columns={"DATE": "Date", series_id: name})
        .set_index("Date")
        .sort_index()
        .dropna()
    )
    return df


def _fallback_macro_data(country: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Offline / local CSV fallback for macro data.

    Expects CSVs with columns: Date, Value in data/macro.
    """
    if country == "United Kingdom":
        cpi = _load_from_csv("cpi_uk.csv", "CPI")
        unemp = _load_from_csv("unemployment_uk.csv", "Unemployment")
        base_rate = _load_from_csv("base_rate_uk.csv", "BaseRate")
    elif country == "United States":
        cpi = _load_from_csv("cpi_us.csv", "CPI")
        unemp = _load_from_csv("unemployment_us.csv", "Unemployment")
        base_rate = _load_from_csv("base_rate_us.csv", "BaseRate")
    elif country == "Euro Area":
        cpi = _load_from_csv("cpi_ea.csv", "CPI")
        unemp = _load_from_csv("unemployment_ea.csv", "Unemployment")
        base_rate = _load_from_csv("base_rate_ea.csv", "BaseRate")
    else:
        # Default to UK if something unexpected slips through
        cpi = _load_from_csv("cpi_uk.csv", "CPI")
        unemp = _load_from_csv("unemployment_uk.csv", "Unemployment")
        base_rate = _load_from_csv("base_rate_uk.csv", "BaseRate")

    return cpi, unemp, base_rate



@st.cache_data
def load_macro_data(country: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load CPI, unemployment and policy rate series for a country.

    - First choice: live FRED (for US/UK/Euro if available).
    - Fallback: local CSVs per country in data/macro.

    Returns (cpi_df, unemployment_df, rate_df, source_label).
    """

    # Always have a local fallback available
    # (this will raise if the CSVs are missing or malformed)
    try:
        local_cpi, local_unemp, local_rate = _fallback_macro_data(country)
    except Exception as e:
        # If even the local files fail, propagate the error
        raise RuntimeError(f"Local CSV fallback failed for {country}: {e}")

    # Try live FRED where configured
    if country in MACRO_SERIES:
        series_map = MACRO_SERIES[country]
        try:
            cpi = fetch_macro_series(series_map["cpi"], "CPI")
            unemp = fetch_macro_series(series_map["unemployment"], "Unemployment")
            base_rate = fetch_macro_series(series_map["rate"], "BaseRate")
            return cpi, unemp, base_rate, "Live FRED"
        except Exception:
            # FRED failed – fall back to local CSV for that country
            return local_cpi, local_unemp, local_rate, f"Local CSV fallback ({country})"

    # If country not in MACRO_SERIES, just use local CSV
    return local_cpi, local_unemp, local_rate, f"Local CSV ({country})"




def add_macro_features(
    cpi_df: pd.DataFrame,
    unemp_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cpi = cpi_df.copy()
    unemp = unemp_df.copy()

    cpi["CPI_YoY"] = cpi["CPI"].pct_change(periods=12) * 100
    unemp["Unemployment_YoY"] = unemp["Unemployment"].pct_change(periods=12) * 100

    return cpi, unemp


def classify_inflation(yoy: float) -> str:
    if pd.isna(yoy):
        return "not enough data"
    if yoy < 1:
        return "low"
    elif yoy < 3:
        return "moderate"
    else:
        return "elevated"


def classify_unemployment(level: float) -> str:
    if pd.isna(level):
        return "not enough data"
    if level < 4:
        return "low"
    elif level < 6:
        return "normal"
    else:
        return "high"


def macro_summary_text(
    cpi_df: pd.DataFrame,
    unemp_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    country: str,
    source: str,
) -> str:
    latest_idx = min(cpi_df.index[-1], unemp_df.index[-1], rate_df.index[-1])
    latest_date = latest_idx.date()

    latest_cpi = cpi_df["CPI"].loc[latest_idx]
    prev_cpi = cpi_df["CPI"].iloc[-2]
    latest_cpi_yoy = cpi_df["CPI_YoY"].loc[latest_idx]

    latest_unemp = unemp_df["Unemployment"].loc[latest_idx]
    prev_unemp = unemp_df["Unemployment"].iloc[-2]
    latest_unemp_yoy = unemp_df["Unemployment_YoY"].loc[latest_idx]

    latest_rate = rate_df["BaseRate"].iloc[-1]
    prev_rate = rate_df["BaseRate"].iloc[-2]
    rate_change = latest_rate - prev_rate

    def direction(latest: float, previous: float) -> str:
        if latest > previous:
            return "rising"
        elif latest < previous:
            return "falling"
        else:
            return "unchanged"

    cpi_dir = direction(latest_cpi, prev_cpi)
    unemp_dir = direction(latest_unemp, prev_unemp)
    rate_dir = direction(latest_rate, prev_rate)

    inflation_regime = classify_inflation(latest_cpi_yoy)
    unemployment_regime = classify_unemployment(latest_unemp)

    summary = f"""
**{country} Macro Summary** (sample ending {latest_date} – {source})

- CPI level: {latest_cpi:.2f} | YoY: {latest_cpi_yoy:.2f}% | Direction: {cpi_dir} | Regime: {inflation_regime}
- Unemployment: {latest_unemp:.2f}% | YoY: {latest_unemp_yoy:.2f}% | Direction: {unemp_dir} | Regime: {unemployment_regime}
- Policy rate: {latest_rate:.2f}% | Change vs last: {rate_change:+.2f} | Direction: {rate_dir}
"""
    return summary



def infer_macro_state(
    cpi_df: pd.DataFrame,
    unemp_df: pd.DataFrame,
    rate_df: pd.DataFrame,
) -> Dict[str, Any]:
    latest_idx = min(cpi_df.index[-1], unemp_df.index[-1], rate_df.index[-1])

    latest_cpi_yoy = cpi_df.loc[latest_idx, "CPI_YoY"]
    latest_unemp = unemp_df.loc[latest_idx, "Unemployment"]

    latest_rate = rate_df["BaseRate"].iloc[-1]
    prev_rate = rate_df["BaseRate"].iloc[-2]

    if latest_rate > prev_rate:
        rate_direction = "rising"
    elif latest_rate < prev_rate:
        rate_direction = "falling"
    else:
        rate_direction = "flat"

    return {
        "inflation_yoy": latest_cpi_yoy,
        "unemployment": latest_unemp,
        "rate_level": latest_rate,
        "rate_direction": rate_direction,
        "inflation_regime": classify_inflation(latest_cpi_yoy),
        "unemployment_regime": classify_unemployment(latest_unemp),
    }


# -----------------------
# Company financials via yfinance
# -----------------------

@st.cache_data
def load_company_financials(ticker: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)

    # Financial statements (annual)
    income = tk.financials
    balance = tk.balance_sheet
    cashflow = tk.cashflow

    # Info
    try:
        info = tk.info
    except Exception:
        info = {}
    if not info:
        try:
            info = tk.get_info()
        except Exception:
            info = {}

    return {
        "ticker": ticker,
        "info": info,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }


def add_kpis(fin: Dict[str, Any]) -> pd.DataFrame:
    income = fin["income"]
    balance = fin["balance"]
    cashflow = fin["cashflow"]

    # If no income statement, we cannot build KPIs
    if income is None or income.empty:
        return pd.DataFrame()

    years = list(income.columns)

    # Revenue and Net income labels
    rev_label = _find_row_label(
        income,
        [
            "Total Revenue",
            "TotalRevenue",
            "Revenue",
            "Total net sales",
            "Total operating revenues",
        ],
    )
    ni_label = _find_row_label(
        income,
        [
            "Net Income",
            "NetIncome",
            "Net income applicable to common shares",
            "Net income available to common stockholders",
        ],
    )

    # Equity label (balance sheet)
    eq_label = _find_equity_label(balance) if balance is not None and not balance.empty else None

    # Debt labels (balance sheet)
    debt_labels = [
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Short Long Term Debt",
        "Current Portion of Long Term Debt",
        "Short-term borrowings",
        "Current debt",
        "Long-term debt",
    ]

    # Operating cash flow and Capex labels (cashflow)
    if cashflow is not None and not cashflow.empty:
        ocf_label = _find_row_label(
            cashflow,
            [
                "Total Cash From Operating Activities",
                "NetCashProvidedByUsedInOperatingActivities",
                "Net cash provided by operating activities",
                "Net cash provided by (used in) operating activities",
                "Operating Cash Flow",
            ],
        )
        capex_label = _find_row_label(
            cashflow,
            [
                "Capital Expenditures",
                "Capital expenditure",
                "Capital expenditures",
                "Purchase of property plant and equipment",
                "Purchase of property, plant and equipment",
                "Purchase of property and equipment",
                "Purchase of fixed assets",
                "Additions to property plant and equipment",
            ],
        )
    else:
        ocf_label = None
        capex_label = None

    rows: List[Dict[str, Any]] = []

    for col in years:
        year_str = str(col)

        # Revenue and net income
        rev = _to_float(income.loc[rev_label, col]) if rev_label in income.index else math.nan
        ni = _to_float(income.loc[ni_label, col]) if ni_label in income.index else math.nan

        # Equity
        if balance is not None and not balance.empty and eq_label and eq_label in balance.index:
            eq = _to_float(balance.loc[eq_label, col])
        else:
            eq = math.nan

        # Debt (sum of any available debt rows)
        debt = 0.0
        if balance is not None and not balance.empty:
            for dl in debt_labels:
                if dl in balance.index:
                    debt_val = _to_float(balance.loc[dl, col])
                    if not math.isnan(debt_val):
                        debt += debt_val
        if debt == 0:
            debt = math.nan

        # Operating cash flow
        if cashflow is not None and not cashflow.empty and ocf_label and ocf_label in cashflow.index:
            ocf = _to_float(cashflow.loc[ocf_label, col])
        else:
            ocf = math.nan

        # Capex
        if cashflow is not None and not cashflow.empty and capex_label and capex_label in cashflow.index:
            capex = _to_float(cashflow.loc[capex_label, col])
        else:
            capex = math.nan

        # Ratios
        net_margin = (
            ni / rev * 100
            if rev not in (0, math.nan) and not math.isnan(rev) and not math.isnan(ni)
            else math.nan
        )
        roe = (
            ni / eq * 100
            if eq not in (0, math.nan) and not math.isnan(eq) and not math.isnan(ni)
            else math.nan
        )
        debt_to_equity = (
            debt / eq
            if eq not in (0, math.nan) and not math.isnan(eq) and not math.isnan(debt)
            else math.nan
        )
        cash_conv = (
            ocf / ni * 100
            if ni not in (0, math.nan) and not math.isnan(ni) and not math.isnan(ocf)
            else math.nan
        )
        capex_to_rev = (
            capex / rev * 100
            if rev not in (0, math.nan) and not math.isnan(rev) and not math.isnan(capex)
            else math.nan
        )

        rows.append(
            {
                "Year": year_str,
                "Revenue": rev,
                "NetIncome": ni,
                "NetMarginPct": net_margin,
                "Equity": eq,
                "ROE_Pct": roe,
                "Debt": debt,
                "DebtToEquity": debt_to_equity,
                "OperatingCashFlow": ocf,
                "CashConversionPct": cash_conv,
                "Capex": capex,
                "CapexToRevenuePct": capex_to_rev,
            }
        )

    df = pd.DataFrame(rows).set_index("Year").sort_index()
    return df



def equity_snapshot_text(ticker: str, fin_kpis: pd.DataFrame) -> str:
    # Prefer the latest year where CapexToRevenuePct is available; otherwise use the last row
    if "CapexToRevenuePct" in fin_kpis.columns:
        non_nan = fin_kpis.dropna(subset=["CapexToRevenuePct"], how="all")
        latest = non_nan.iloc[-1] if not non_nan.empty else fin_kpis.iloc[-1]
    else:
        latest = fin_kpis.iloc[-1]

    rev = latest.get("Revenue", math.nan)
    ni = latest.get("NetIncome", math.nan)
    net_margin = latest.get("NetMarginPct", math.nan)
    roe = latest.get("ROE_Pct", math.nan)
    d_to_e = latest.get("DebtToEquity", math.nan)
    cash_conv = latest.get("CashConversionPct", math.nan)
    capex_rev = latest.get("CapexToRevenuePct", math.nan)

    text = f"""
**Equity snapshot for {ticker} (latest usable year in sample)**

- Revenue: {rev:,.0f}
- Net income: {ni:,.0f}
- Net margin: {net_margin:.2f}% 
- ROE: {roe:.2f}%
- Debt to equity: {d_to_e:.2f}
- Cash conversion (OCF / NI): {cash_conv:.2f}%
- Capex / revenue: {capex_rev:.2f}%
"""
    return text



def combined_view_text(ticker: str, fin_kpis: pd.DataFrame, macro_state: Dict[str, Any]) -> str:
    latest = fin_kpis.iloc[-1]

    net_margin = latest.get("NetMarginPct", math.nan)
    roe = latest.get("ROE_Pct", math.nan)

    infl_reg = macro_state.get("inflation_regime", "unknown")
    unemp_reg = macro_state.get("unemployment_regime", "unknown")
    rate_dir = macro_state.get("rate_direction", "unknown")

    text = f"""
**Macro-aware view for {ticker}**

- Profitability now: net margin {net_margin:.2f}%, ROE {roe:.2f}%.
- Macro backdrop: inflation regime is **{infl_reg}**, unemployment is **{unemp_reg}**, policy rates are **{rate_dir}**.

This is not a forecast, but a way to frame whether the current equity performance is happening in an easy or a tough macro environment.
"""
    return text


def build_portfolio_table(tickers: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for t in tickers:
        t_clean = t.upper().strip()
        if not t_clean:
            continue

        try:
            # Load financials and KPIs
            fin = load_company_financials(t_clean)
            fin_kpis = add_kpis(fin)
            if fin_kpis.empty:
                raise ValueError("No financial KPIs available")

            # Prefer the latest year where CapexToRevenuePct is available; otherwise use the last row
            if "CapexToRevenuePct" in fin_kpis.columns:
                non_nan = fin_kpis.dropna(subset=["CapexToRevenuePct"], how="all")
                latest = non_nan.iloc[-1] if not non_nan.empty else fin_kpis.iloc[-1]
            else:
                latest = fin_kpis.iloc[-1]

            # Robust info lookup for Sector / Country / Name
            info = fin.get("info") or {}
            if not info:
                # Fallback: query yfinance again specifically for metadata
                try:
                    info = yf.Ticker(t_clean).get_info()
                except Exception:
                    info = {}

            name = info.get("shortName") or info.get("longName") or t_clean
            sector = info.get("sector") or info.get("industry") or ""
            country = info.get("country") or info.get("countryISO") or ""

            row = {
                "Ticker": t_clean,
                "Name": name,
                "Sector": sector,
                "Country": country,
                "Revenue": latest.get("Revenue", math.nan),
                "NetIncome": latest.get("NetIncome", math.nan),
                "NetMarginPct": latest.get("NetMarginPct", math.nan),
                "Equity": latest.get("Equity", math.nan),
                "ROE_Pct": latest.get("ROE_Pct", math.nan),
                "Debt": latest.get("Debt", math.nan),
                "DebtToEquity": latest.get("DebtToEquity", math.nan),
                "OperatingCashFlow": latest.get("OperatingCashFlow", math.nan),
                "CashConversionPct": latest.get("CashConversionPct", math.nan),
                "Capex": latest.get("Capex", math.nan),
                "CapexToRevenuePct": latest.get("CapexToRevenuePct", math.nan),
            }
            rows.append(row)

        except Exception as e:
            failures.append((t_clean, str(e)))

    df = pd.DataFrame(rows)

    if not df.empty:
        desired_cols = [
            "Name",
            "Sector",
            "Country",
            "Revenue",
            "NetIncome",
            "NetMarginPct",
            "Equity",
            "ROE_Pct",
            "Debt",
            "DebtToEquity",
            "OperatingCashFlow",
            "CashConversionPct",
            "Capex",
            "CapexToRevenuePct",
        ]
        for c in desired_cols:
            if c not in df.columns:
                df[c] = math.nan

        df = df.set_index("Ticker")
        df = df[desired_cols]

    return df, failures




# -----------------------
# Streamlit layout
# -----------------------

st.set_page_config(
    page_title="Macro–Equity Dashboard",
    layout="wide",
)

st.title("Macro–Equity Dashboard")
st.caption("Always-on macro context plus live yfinance fundamentals for megacaps and your own watchlist.")

# Sidebar mode selection
st.sidebar.header("Settings")

country = st.sidebar.selectbox("Macro country", list(MACRO_SERIES.keys()), index=1)

mode = st.sidebar.radio(
    "Mode",
    ["Portfolio (Top 10)", "Single company"],
)

default_selection = st.sidebar.multiselect(
    "Pick firms to analyze (portfolio mode)", PORTFOLIO_TICKERS, default=PORTFOLIO_TICKERS
)
extra_tickers = st.sidebar.text_input(
    "Extra tickers (comma separated)",
    help="Add any additional yfinance tickers to include in the portfolio table.",
)

# Load macro
# Load macro
try:
    cpi_raw, unemp_raw, rate_raw, source_label = load_macro_data(country)
    cpi, unemp = add_macro_features(cpi_raw, unemp_raw)
    macro_state = infer_macro_state(cpi, unemp, rate_raw)
    macro_summary = macro_summary_text(cpi, unemp, rate_raw, country, source_label)
except Exception as e:
    st.error(f"Error loading macro data: {e}")
    st.stop()

# Macro overview at top
st.subheader("Macro overview")
st.markdown(f"**Data source:** {source_label}")
st.markdown(macro_summary)

macro_cols = st.columns(3)
with macro_cols[0]:
    st.markdown("**CPI (last 36 observations)**")
    st.line_chart(cpi.tail(36)[["CPI"]])
with macro_cols[1]:
    st.markdown("**Unemployment (all)**")
    st.line_chart(unemp[["Unemployment"]])
with macro_cols[2]:
    st.markdown("**Policy rate (all)**")
    st.line_chart(rate_raw[["BaseRate"]])

st.markdown("---")

# -----------------------
# Mode: Single company
# -----------------------
if mode == "Single company":
    st.subheader("Single company analysis")

    preset = st.selectbox("Quick pick (top firms)", PORTFOLIO_TICKERS, index=1)
    ticker_input = st.text_input(
        "Ticker (yfinance symbol)",
        value=preset,
        help="Any ticker supported by yfinance",
    ).upper().strip()

    if ticker_input:
        try:
            fin = load_company_financials(ticker_input)
            fin_kpis = add_kpis(fin)

            if fin_kpis.empty:
                st.error("No financial data available for this ticker.")
            else:
                st.markdown("### Financial KPIs (latest years)")
                st.dataframe(fin_kpis.round(2))

                st.markdown("### Equity snapshot")
                st.markdown(equity_snapshot_text(ticker_input, fin_kpis))

                st.markdown("### Macro aware view")
                st.markdown(combined_view_text(ticker_input, fin_kpis, macro_state))

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Revenue by year**")
                    st.bar_chart(fin_kpis["Revenue"])
                with c2:
                    st.markdown("**Net margin (%) by year**")
                    st.bar_chart(fin_kpis["NetMarginPct"])

        except Exception as e:
            st.error(f"Could not load data for {ticker_input}: {e}")

# -----------------------
# Mode: Portfolio
# -----------------------
else:
    st.subheader("Portfolio view")

    tickers = [t.upper().strip() for t in default_selection if t.strip()]
    if extra_tickers:
        tickers.extend([t.strip().upper() for t in extra_tickers.split(",") if t.strip()])

    tickers = sorted(set(tickers))

    if not tickers:
        st.warning("Please choose at least one ticker to build the portfolio table.")
        st.stop()

    st.markdown(f"Default portfolio tickers: `{', '.join(PORTFOLIO_TICKERS)}`")
    st.markdown(f"Active portfolio tickers ({len(tickers)}): `{', '.join(tickers)}`")

    portfolio_df, failures = build_portfolio_table(tickers)

    if portfolio_df.empty:
        st.error("Could not build portfolio table (no tickers succeeded).")
    else:
        st.markdown("### Portfolio fundamentals (latest year per company)")
        st.dataframe(portfolio_df.round(2))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Revenue (latest year)**")
            st.bar_chart(portfolio_df["Revenue"])
        with c2:
            st.markdown("**ROE (%) (latest year)**")
            st.bar_chart(portfolio_df["ROE_Pct"])

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Net margin (%) (latest year)**")
            st.bar_chart(portfolio_df["NetMarginPct"])
        with c4:
            st.markdown("**Debt to equity (latest year)**")
            st.bar_chart(portfolio_df["DebtToEquity"])

        if failures:
            fail_str = ", ".join([f"{t} ({msg})" for t, msg in failures])
            st.warning(f"Some tickers failed to load: {fail_str}")
