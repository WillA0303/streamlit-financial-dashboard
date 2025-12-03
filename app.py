##############################################
# app.py – Macro–Equity Dashboard (Streamlit)
##############################################

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping
import io
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

CONFIG: Dict[str, Any] = {
    "macro_series": {
        "United States": {
            "cpi": "CPIAUCSL",  # CPI for All Urban Consumers: All Items
            "unemployment": "UNRATE",  # Unemployment Rate
            "rate": "FEDFUNDS",  # Effective Federal Funds Rate
        },
        "United Kingdom": {
            "cpi": "GBRCPIALLMINMEI",  # CPI All Items
            "unemployment": "LRHUTTTTGBQ156S",  # Harmonized unemployment rate
            "rate": "BOEBASE",  # Bank of England Bank Rate
        },
        "Euro Area": {
            "cpi": "CP0000EZ19M086NEST",  # Euro Area CPI All Items
            "unemployment": "LRHUTTTTEZM156S",  # Harmonized unemployment rate
            "rate": "ECBDFR",  # ECB Deposit Facility Rate
        },
    },
    "portfolio_tickers": [
        "NVDA",
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "AVGO",
        "META",
        "2222.SR",
        "TSM",
        "TSLA",
    ],
    "macro_horizons": {
        "2Y": "730D",
        "5Y": "1825D",
        "Full": None,
    },
}

MACRO_SERIES: Dict[str, Dict[str, str]] = CONFIG["macro_series"]
PORTFOLIO_TICKERS: List[str] = CONFIG["portfolio_tickers"]

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


def _format_metric_value(value: Any, fmt: str, placeholder: str = "N/A") -> str:
    try:
        fval = float(value)
        if math.isnan(fval):
            return placeholder
        return fmt.format(fval)
    except Exception:
        return placeholder


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
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Timed out fetching {series_id} from FRED") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed for {series_id}: {e}") from e

    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
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
    elif level < 7:
        return "moderate"
    else:
        return "elevated"


def summarize_macro_state(cpi: pd.DataFrame, unemp: pd.DataFrame, rate: pd.DataFrame) -> Dict[str, Any]:
    cpi_aligned = cpi.last("5Y").copy()
    unemp_aligned = unemp.last("5Y").copy()
    rate_aligned = rate.last("5Y").copy()

    if "CPI_YoY" not in cpi_aligned.columns and "CPI" in cpi_aligned.columns:
        cpi_aligned["CPI_YoY"] = cpi_aligned["CPI"].pct_change(periods=12) * 100

    latest_cpi_yoy = cpi_aligned["CPI_YoY"].iloc[-1]
    latest_unemp = unemp_aligned["Unemployment"].iloc[-1]

    if len(rate_aligned) >= 2:
        latest_rate = rate_aligned["BaseRate"].iloc[-1]
        prev_rate = rate_aligned["BaseRate"].iloc[-2]
    else:
        latest_rate = rate_aligned["BaseRate"].iloc[-1]
        prev_rate = latest_rate

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

def _load_ticker_info(tk: yf.Ticker) -> Dict[str, Any]:
    """Best-effort info fetch that tolerates yfinance quirks."""

    def _normalize(candidate: Any) -> Dict[str, Any]:
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, Mapping):
            return dict(candidate)
        try:
            return dict(candidate)
        except Exception:
            return {}

    def _candidate_info() -> Dict[str, Any]:
        try:
            return _normalize(tk.info)
        except Exception:
            return {}

    def _candidate_get_info() -> Dict[str, Any]:
        try:
            return _normalize(tk.get_info())
        except Exception:
            return {}

    def _candidate_fast_info() -> Dict[str, Any]:
        try:
            return _normalize(tk.fast_info)
        except Exception:
            return {}

    def _candidate_get_fast_info() -> Dict[str, Any]:
        try:
            return _normalize(tk.get_fast_info())
        except Exception:
            return {}

    merged: Dict[str, Any] = {}
    for getter in (
        _candidate_info,
        _candidate_get_info,
        _candidate_fast_info,
        _candidate_get_fast_info,
    ):
        candidate = getter()
        if not candidate:
            continue

        for key, value in candidate.items():
            if key not in merged or merged.get(key) in (None, "", math.nan):
                merged[key] = value

    return merged


@st.cache_data
def load_company_financials(ticker: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)

    # Financial statements (annual)
    income = tk.financials
    balance = tk.balance_sheet
    cashflow = tk.cashflow

    # Info
    info = _load_ticker_info(tk)

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
                "CapitalExpenditures",
                "PurchaseOfPropertyPlantAndEquipment",
                "Capital expenditure",
            ],
        )
    else:
        ocf_label = None
        capex_label = None

    # Debt value (prefer long-term + current)
    debt_val = None
    if balance is not None and not balance.empty:
        for lbl in debt_labels:
            val = balance.loc[lbl, years[0]] if lbl in balance.index else None
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                debt_val = val
                break

    data = {
        "Revenue": [income.loc[rev_label, y] if rev_label in income.index else math.nan for y in years],
        "NetIncome": [income.loc[ni_label, y] if ni_label in income.index else math.nan for y in years],
        "Equity": [balance.loc[eq_label, y] if eq_label in balance.index else math.nan for y in years]
        if balance is not None and not balance.empty
        else [math.nan for _ in years],
        "Debt": [debt_val for _ in years],
        "OperatingCashFlow": [
            cashflow.loc[ocf_label, y] if cashflow is not None and ocf_label in cashflow.index else math.nan
            for y in years
        ],
        "Capex": [cashflow.loc[capex_label, y] if cashflow is not None and capex_label in cashflow.index else math.nan for y in years],
    }

    df = pd.DataFrame(data, index=years)

    df["NetMarginPct"] = (df["NetIncome"] / df["Revenue"]) * 100
    df["ROE_Pct"] = (df["NetIncome"] / df["Equity"]) * 100
    df["DebtToEquity"] = df["Debt"] / df["Equity"]
    df["CashConversionPct"] = (df["OperatingCashFlow"] / df["NetIncome"]) * 100
    df["CapexToRevenuePct"] = (df["Capex"] / df["Revenue"]) * 100

    df.index = df.index.astype(str)

    return df


def valuation_from_info(info: Dict[str, Any], latest_revenue: float) -> Dict[str, float]:
    market_cap = _to_float(info.get("marketCap"))
    trailing_pe = _to_float(info.get("trailingPE"))
    forward_pe = _to_float(info.get("forwardPE"))
    price_to_book = _to_float(info.get("priceToBook"))
    enterprise_value = _to_float(info.get("enterpriseValue"))
    ev_to_sales = _to_float(info.get("enterpriseToRevenue"))
    fcf = _to_float(info.get("freeCashflow"))

    fcf_yield = (fcf / market_cap) * 100 if market_cap not in (0, math.nan) and not math.isnan(market_cap) else math.nan

    return {
        "MarketCap": market_cap,
        "TrailingPE": trailing_pe,
        "ForwardPE": forward_pe,
        "PriceToBook": price_to_book,
        "EnterpriseValue": enterprise_value,
        "EV_to_Sales": ev_to_sales,
        "FreeCashFlow": fcf,
        "FCF_YieldPct": fcf_yield,
    }


def quick_dcf_value(
    fin_kpis: pd.DataFrame,
    info: Dict[str, Any],
    sales_growth_pct: float,
    net_margin_pct: float,
    cost_of_equity_pct: float,
) -> Dict[str, float]:
    latest_revenue = fin_kpis.iloc[-1].get("Revenue", math.nan)
    shares_outstanding = _to_float(info.get("sharesOutstanding"))

    forward_revenue = latest_revenue * (1 + sales_growth_pct / 100)
    forward_income = forward_revenue * (net_margin_pct / 100)
    discount_rate = cost_of_equity_pct / 100

    if math.isnan(forward_income) or discount_rate <= 0:
        return {}

    equity_value = forward_income / discount_rate
    per_share = (
        equity_value / shares_outstanding
        if shares_outstanding not in (0, math.nan) and not math.isnan(shares_outstanding)
        else math.nan
    )

    return {
        "forward_revenue": forward_revenue,
        "forward_income": forward_income,
        "equity_value": equity_value,
        "per_share_value": per_share,
    }


def price_vs_macro(ticker: str, cpi_df: pd.DataFrame) -> pd.DataFrame:
    try:
        price_hist = yf.download(ticker, period="10y", progress=False)
    except Exception:
        return pd.DataFrame()

    if price_hist is None or price_hist.empty:
        return pd.DataFrame()

    prices = price_hist.get("Adj Close")
    if prices is None or prices.empty:
        return pd.DataFrame()

    rolling_return = prices.pct_change(252) * 100
    rolling_return.name = "PriceReturn12M"

    cpi_resampled = cpi_df.copy()
    if "CPI_YoY" not in cpi_resampled.columns and "CPI" in cpi_resampled.columns:
        cpi_resampled["CPI_YoY"] = cpi_resampled["CPI"].pct_change(periods=12) * 100

    try:
        cpi_resampled.index = pd.to_datetime(cpi_resampled.index)
        cpi_resampled = cpi_resampled[["CPI_YoY"]].resample("B").ffill()
    except Exception:
        return pd.DataFrame()

    combined = pd.concat([rolling_return, cpi_resampled["CPI_YoY"]], axis=1).dropna()

    return combined


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
                    info = _load_ticker_info(yf.Ticker(t_clean))
                except Exception:
                    info = {}

            name = info.get("shortName") or info.get("longName") or t_clean
            sector = info.get("sector") or info.get("industry") or "Unknown"
            country = info.get("country") or info.get("countryISO") or "Unknown"

            valuation = valuation_from_info(info, latest.get("Revenue", math.nan))

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
                "MarketCap": valuation.get("MarketCap", math.nan),
                "TrailingPE": valuation.get("TrailingPE", math.nan),
                "ForwardPE": valuation.get("ForwardPE", math.nan),
                "PriceToBook": valuation.get("PriceToBook", math.nan),
                "EnterpriseValue": valuation.get("EnterpriseValue", math.nan),
                "EV_to_Sales": valuation.get("EV_to_Sales", math.nan),
                "FreeCashFlow": valuation.get("FreeCashFlow", math.nan),
                "FCF_YieldPct": valuation.get("FCF_YieldPct", math.nan),
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
            "MarketCap",
            "TrailingPE",
            "ForwardPE",
            "PriceToBook",
            "EnterpriseValue",
            "EV_to_Sales",
            "FreeCashFlow",
            "FCF_YieldPct",
        ]
        for c in desired_cols:
            if c not in df.columns:
                df[c] = math.nan

        df = df.set_index("Ticker")
        df = df[desired_cols]

    return df, failures


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(
    page_title="Macro–Equity Dashboard",
    layout="wide",
    page_icon="📈",
)

st.title("Macro–Equity Dashboard")

country = st.selectbox("Select country for macro overview", list(MACRO_SERIES.keys()))

try:
    cpi_df, unemp_df, rate_df, source_label = load_macro_data(country)
except Exception as e:
    st.error(f"Could not load macro data: {e}")
    st.stop()

macro_state = summarize_macro_state(cpi_df, unemp_df, rate_df)
macro_summary = f"""
**Inflation:** {macro_state['inflation_regime']} (YoY {macro_state['inflation_yoy']:.2f}%)
**Unemployment:** {macro_state['unemployment_regime']} (Level {macro_state['unemployment']:.2f}%)
**Rates:** {macro_state['rate_direction']} (Latest {macro_state['rate_level']:.2f}%)
"""

cpi, unemp = add_macro_features(cpi_df, unemp_df)
rate = rate_df.copy()

# Limit to selected horizon
horizon_label = st.radio("Horizon", list(CONFIG["macro_horizons"].keys()), index=0)
horizon_period = CONFIG["macro_horizons"][horizon_label]

if horizon_period:
    cpi_plot = cpi.last(horizon_period)
    unemp_plot = unemp.last(horizon_period)
    rate_plot = rate.last(horizon_period)
else:
    cpi_plot = cpi
    unemp_plot = unemp
    rate_plot = rate

tab_macro, tab_single, tab_portfolio = st.tabs(
    ["Macro overview", "Single company", "Portfolio"]
)

with tab_macro:
    st.subheader("Macro overview")
    st.markdown(f"**Data source:** {source_label}")
    st.success(macro_summary)

    regime_cols = st.columns(3)
    regime_cols[0].metric(
        "Inflation regime",
        macro_state["inflation_regime"],
        f"YoY {macro_state['inflation_yoy']:.2f}%",
    )
    regime_cols[1].metric(
        "Unemployment regime",
        macro_state["unemployment_regime"],
        f"Level {macro_state['unemployment']:.2f}%",
    )
    regime_cols[2].metric(
        "Rate direction",
        macro_state["rate_direction"],
        f"Latest {macro_state['rate_level']:.2f}%",
    )

    st.markdown(macro_summary)

    macro_cols = st.columns(3)
    with macro_cols[0]:
        st.markdown("**CPI**")
        st.line_chart(cpi_plot[["CPI"]])
    with macro_cols[1]:
        st.markdown("**Unemployment**")
        st.line_chart(unemp_plot[["Unemployment"]])
    with macro_cols[2]:
        st.markdown("**Policy rate**")
        st.line_chart(rate_plot[["BaseRate"]])

    macro_download_df = pd.concat(
        [cpi_plot[["CPI"]], unemp_plot[["Unemployment"]], rate_plot[["BaseRate"]]],
        axis=1,
    )
    st.download_button(
        "Download macro series (CSV)",
        data=macro_download_df.to_csv().encode("utf-8"),
        file_name=f"macro_series_{country.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

with tab_single:
    st.subheader("Single company analysis")

    if "single_ticker" not in st.session_state:
        st.session_state["single_ticker"] = PORTFOLIO_TICKERS[1]

    def _sync_single_preset() -> None:
        st.session_state["single_ticker"] = st.session_state.get("single_preset", "").upper().strip()

    preset = st.selectbox(
        "Quick pick (top firms)",
        PORTFOLIO_TICKERS,
        index=1,
        key="single_preset",
        on_change=_sync_single_preset,
    )
    ticker_input = st.text_input(
        "Ticker (yfinance symbol)",
        value=st.session_state.get("single_ticker", preset),
        help="Any ticker supported by yfinance",
        key="single_ticker",
    ).upper().strip()

    growth = st.slider("Sales growth (%)", min_value=-10.0, max_value=30.0, value=5.0, step=0.5)
    margin = st.slider("Net margin (%)", min_value=-20.0, max_value=50.0, value=15.0, step=0.5)
    cost_equity = st.slider("Cost of equity (%)", min_value=2.0, max_value=20.0, value=8.0, step=0.5)

    if ticker_input:
        try:
            fin = load_company_financials(ticker_input)
            fin_kpis = add_kpis(fin)

            if fin_kpis.empty:
                st.warning("No annual financials found for this ticker. Try another symbol.")
            else:
                info = fin.get("info") or {}
                valuations = valuation_from_info(info, fin_kpis.iloc[-1].get("Revenue", math.nan))

                st.markdown("### Financial KPIs (latest years)")
                st.dataframe(fin_kpis.round(2))

                val_cols = st.columns(4)
                val_cols[0].metric(
                    "Market cap",
                    _format_metric_value(valuations["MarketCap"], "{:,.0f}"),
                )
                val_cols[1].metric(
                    "Trailing PE", _format_metric_value(valuations["TrailingPE"], "{:.2f}")
                )
                val_cols[2].metric(
                    "PB", _format_metric_value(valuations["PriceToBook"], "{:.2f}")
                )
                val_cols[3].metric(
                    "FCF yield (%)",
                    _format_metric_value(valuations["FCF_YieldPct"], "{:.2f}"),
                )

                dcf_result = quick_dcf_value(fin_kpis, info, growth, margin, cost_equity)
                if dcf_result:
                    st.markdown("### Simple one-period DCF scenario")
                    dcf_cols = st.columns(3)
                    dcf_cols[0].metric("Forward revenue", f"{dcf_result['forward_revenue']:,.0f}")
                    dcf_cols[1].metric("Forward income", f"{dcf_result['forward_income']:,.0f}")
                    dcf_cols[2].metric("Equity value", f"{dcf_result['equity_value']:,.0f}")
                    if not math.isnan(dcf_result.get("per_share_value", math.nan)):
                        st.info(f"Implied value per share: {dcf_result['per_share_value']:.2f}")
                else:
                    st.info("Adjust the sliders to compute a quick DCF view.")

                st.markdown("### Equity snapshot")
                st.markdown(equity_snapshot_text(ticker_input, fin_kpis))

                st.markdown("### Macro aware view")
                st.markdown(combined_view_text(ticker_input, fin_kpis, macro_state))

                perf_cols = st.columns(2)
                with perf_cols[0]:
                    st.markdown("**Revenue by year**")
                    st.bar_chart(fin_kpis["Revenue"])
                with perf_cols[1]:
                    st.markdown("**Net margin (%) by year**")
                    st.bar_chart(fin_kpis["NetMarginPct"])

                st.markdown("### Relative performance vs macro")
                price_macro = price_vs_macro(ticker_input, cpi)
                if price_macro.empty:
                    st.info("Not enough price history to plot performance vs macro.")
                else:
                    st.line_chart(price_macro)

        except Exception as e:
            st.error(f"Could not load data for {ticker_input}: {e}")

with tab_portfolio:
    st.subheader("Portfolio view")

    default_selection = st.multiselect(
        "Pick firms to analyze (portfolio mode)", PORTFOLIO_TICKERS, default=PORTFOLIO_TICKERS, key="portfolio_defaults"
    )
    extra_tickers = st.text_input(
        "Extra tickers (comma separated)",
        help="Add any additional yfinance tickers to include in the portfolio table.",
        key="portfolio_extra",
    )
    use_valid_tickers = st.checkbox(
        "Use only valid tickers (dedupe & drop blanks)", value=True, key="portfolio_valid"
    )

    tickers = [t.upper().strip() for t in default_selection]
    if extra_tickers:
        tickers.extend([t.strip().upper() for t in extra_tickers.split(",")])

    if use_valid_tickers:
        tickers = [t for t in tickers if t]
        tickers = list(dict.fromkeys(tickers))
    else:
        tickers = [t for t in tickers if t]

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

        st.download_button(
            "Download portfolio table as CSV",
            data=portfolio_df.to_csv().encode("utf-8"),
            file_name="portfolio_fundamentals.csv",
            mime="text/csv",
        )

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
            st.info("Some tickers may have incomplete statements in yfinance; this is not an app error.")
