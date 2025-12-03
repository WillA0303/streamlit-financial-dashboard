  ##############################################
# app.py – Macro–Equity Dashboard (Streamlit)
##############################################

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    # Basic validation
    for df in (cpi_df, unemp_df, rate_df):
        if df is None or df.empty:
            raise ValueError("Macro data is empty; cannot build summary")

    def latest_and_previous(series: pd.Series) -> Tuple[float, float]:
        if len(series) >= 2:
            return series.iloc[-1], series.iloc[-2]
        if len(series) == 1:
            return series.iloc[-1], series.iloc[-1]
        return math.nan, math.nan

    # Use a common monthly index based on CPI and unemployment
    latest_idx = min(cpi_df.index[-1], unemp_df.index[-1])
    latest_date = latest_idx.date()

    # CPI (truncate to dates <= latest_idx)
    cpi_series = cpi_df["CPI"].loc[:latest_idx]
    latest_cpi, prev_cpi = latest_and_previous(cpi_series)
    latest_cpi_yoy = cpi_df["CPI_YoY"].loc[latest_idx]

    # Unemployment (truncate to dates <= latest_idx)
    unemp_series = unemp_df["Unemployment"].loc[:latest_idx]
    latest_unemp, prev_unemp = latest_and_previous(unemp_series)
    latest_unemp_yoy = unemp_df["Unemployment_YoY"].loc[latest_idx]

    # Policy rate: use last two observations up to latest_idx
    rate_series = rate_df["BaseRate"].loc[:latest_idx]
    if len(rate_series) >= 2:
        latest_rate = rate_series.iloc[-1]
        prev_rate = rate_series.iloc[-2]
    elif len(rate_series) == 1:
        latest_rate = rate_series.iloc[-1]
        prev_rate = latest_rate
    else:
        raise ValueError("No rate data available up to latest CPI/unemployment date")
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
    if cpi_df.empty or unemp_df.empty or rate_df.empty:
        raise ValueError("Macro data missing for regime inference")

    # Work off the common monthly index
    latest_idx = min(cpi_df.index[-1], unemp_df.index[-1])

    latest_cpi_yoy = cpi_df.loc[latest_idx, "CPI_YoY"]
    latest_unemp = unemp_df.loc[latest_idx, "Unemployment"]

    # Policy rate: align to latest_idx using last two observations up to that date
    rate_series = rate_df["BaseRate"].loc[:latest_idx]
    if len(rate_series) >= 2:
        latest_rate = rate_series.iloc[-1]
        prev_rate = rate_series.iloc[-2]
    elif len(rate_series) == 1:
        latest_rate = rate_series.iloc[-1]
        prev_rate = latest_rate
    else:
        raise ValueError("No rate data available up to latest CPI/unemployment date")

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
        "prev_rate": prev_rate,
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

    def _safe_df_attr(attr_name: str) -> pd.DataFrame:
        try:
            obj = getattr(tk, attr_name)
        except Exception:
            return pd.DataFrame()
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame) and obj.empty:
            return pd.DataFrame()
        return obj

    def _safe_df_call(method_name: str) -> pd.DataFrame:
        try:
            method = getattr(tk, method_name)
        except Exception:
            return pd.DataFrame()
        try:
            df = method()
        except Exception:
            return pd.DataFrame()
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df

    # Income statement (annual)
    income = _safe_df_attr("financials")
    if income.empty:
        income = _safe_df_attr("income_stmt")
    if income.empty:
        income = _safe_df_call("get_financials")

    # Balance sheet (annual)
    balance = _safe_df_attr("balance_sheet")
    if balance.empty:
        balance = _safe_df_call("get_balance_sheet")

    # Cash flow statement (annual)
    cashflow = _safe_df_attr("cashflow")
    if cashflow.empty:
        cashflow = _safe_df_call("get_cashflow")

    # Company info / metadata
    # Company info / metadata
        # Company info / metadata
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

def valuation_from_info(info: Dict[str, Any], latest_revenue: float) -> Dict[str, float]:
    market_cap = _to_float(info.get("marketCap"))
    enterprise_value = _to_float(info.get("enterpriseValue"))
    trailing_pe = _to_float(info.get("trailingPE"))
    forward_pe = _to_float(info.get("forwardPE"))
    price_to_book = _to_float(info.get("priceToBook"))
    free_cashflow = _to_float(info.get("freeCashflow"))

    ev_to_sales = (
        enterprise_value / latest_revenue
        if latest_revenue not in (0, math.nan) and not math.isnan(latest_revenue)
        and enterprise_value not in (0, math.nan) and not math.isnan(enterprise_value)
        else math.nan
    )
    fcf_yield = (
        free_cashflow / market_cap * 100
        if market_cap not in (0, math.nan) and not math.isnan(market_cap)
        and free_cashflow not in (0, math.nan) and not math.isnan(free_cashflow)
        else math.nan
    )

    return {
        "MarketCap": market_cap,
        "TrailingPE": trailing_pe,
        "ForwardPE": forward_pe,
        "PriceToBook": price_to_book,
        "EnterpriseValue": enterprise_value,
        "EV_to_Sales": ev_to_sales,
        "FreeCashFlow": free_cashflow,
        "FCF_YieldPct": fcf_yield,
    }


def quick_dcf_value(
    fin_kpis: pd.DataFrame,
    info: Dict[str, Any],
    sales_growth_pct: float,
    net_margin_pct: float,
    cost_of_equity_pct: float,
) -> Dict[str, float]:
    if fin_kpis.empty:
        return {}

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
    # Download up to 10 years of daily prices
    price_hist = yf.download(ticker, period="10y", progress=False)
    if price_hist is None or len(price_hist) == 0:
        return pd.DataFrame()

    # Robust selection of a usable price series
    if isinstance(price_hist, pd.DataFrame):
        if "Adj Close" in price_hist.columns:
            prices = price_hist["Adj Close"]
        elif "Close" in price_hist.columns:
            prices = price_hist["Close"]
        else:
            return pd.DataFrame()  # no usable price column
    else:
        # If a Series is returned for some reason, just use it directly
        prices = price_hist

    # 12-month rolling price return in percent
    rolling_return = prices.pct_change(252) * 100

    # Align CPI YoY with business days
    cpi_yoy = cpi_df[["CPI_YoY"]].resample("B").ffill()

    combined = pd.concat(
        [rolling_return.rename("PriceReturn12M"), cpi_yoy["CPI_YoY"]],
        axis=1,
    ).dropna()

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
            st.write("Info keys:", list(info.keys()))

            if not info:
                # Fallback: query yfinance again specifically for metadata
                try:
                    info = yf.Ticker(t_clean).get_info()
                except Exception:
                    info = {}

            name = info.get("shortName") or info.get("longName") or t_clean
            sector = info.get("sector") or info.get("industry") or ""
            country = info.get("country") or info.get("countryISO") or ""

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
# Streamlit layout
# -----------------------
def apply_horizon(df: pd.DataFrame, horizon_key: str) -> pd.DataFrame:
    period = CONFIG["macro_horizons"].get(horizon_key)
    if period:
        try:
            return df.last(period)
        except Exception:
            return df
    return df

st.set_page_config(
    page_title="Macro–Equity Dashboard",
    layout="wide",
)

st.title("Macro–Equity Dashboard")
st.caption("Always-on macro context plus live yfinance fundamentals for megacaps and your own watchlist.")

st.sidebar.header("Settings")
country = st.sidebar.selectbox("Macro country", list(MACRO_SERIES.keys()), index=1)
horizon = st.sidebar.selectbox(
    "Macro horizon", list(CONFIG["macro_horizons"].keys()), index=1
)
with st.sidebar.expander("About this dashboard"):
    st.markdown(
        """
This app combines macroeconomic indicators with live equity fundamentals to create 
a simple, always-updated environment for evaluating companies in their current macro context.

**Macro tab**
- CPI, unemployment, and policy rates for the US, UK, and Euro Area  
- Year-on-year inflation and unemployment dynamics  
- Automatic regime classification (low/moderate/elevated inflation; low/normal/high unemployment)  
- Horizon filter for short-term vs long-term views  
- Downloadable CSV output

**Single company tab**
- yfinance financial statements (income, balance sheet, cash flow)  
- Automatic calculation of KPIs (margins, ROE, leverage, cash conversion, capex intensity)  
- Simple scenario-based DCF using growth, margin, and cost of equity sliders  
- Macro-aware interpretation of the company’s latest fundamentals  
- Price vs inflation chart over the last decade

**Portfolio tab**
- Multi-ticker fundamentals table for comparison  
- Aggregated KPIs for a watchlist  
- Quick visual comparisons across revenue, ROE, margins, and leverage  
- CSV download for further analysis

**Data sources**
- FRED (US macro), ONS (UK inflation & unemployment), Bank of England (Bank Rate)  
- yfinance (equity fundamentals)  
- Local CSV data for extended UK history

**Purpose**
This dashboard is designed as a practical, lightweight research tool and a demonstration
of how macro and company-level data can be combined in a single analytical interface.
It is informative rather than predictive.
"""
    )

try:
    cpi_raw, unemp_raw, rate_raw, source_label = load_macro_data(country)
    cpi, unemp = add_macro_features(cpi_raw, unemp_raw)
    macro_state = infer_macro_state(cpi, unemp, rate_raw)
    macro_summary = macro_summary_text(cpi, unemp, rate_raw, country, source_label)
except Exception as e:
    st.error(f"Error loading macro data: {e}")
    st.stop()

# Apply horizon for plotting and downloads
cpi_plot = apply_horizon(cpi, horizon)
unemp_plot = apply_horizon(unemp, horizon)
rate_plot = apply_horizon(rate_raw, horizon)

macro_regime_label = (
    f"Regime: {macro_state['inflation_regime']} inflation, "
    f"{macro_state['unemployment_regime']} unemployment, "
    f"{macro_state['rate_direction']} rates"
)
tab_macro, tab_single, tab_portfolio = st.tabs(
    ["Macro overview", "Single company", "Portfolio"]
)

with tab_macro:
    st.subheader("Macro overview")
    st.markdown(f"**Data source:** {source_label}")
    st.success(macro_regime_label)

    regime_cols = st.columns(3)
    regime_cols[0].metric(
        "Inflation regime",
        macro_state["inflation_regime"],
        f"YoY {macro_state['inflation_yoy']:.2f}%"
    )
    regime_cols[1].metric(
        "Unemployment regime",
        macro_state["unemployment_regime"],
        f"Level {macro_state['unemployment']:.2f}%"
    )
    regime_cols[2].metric(
        "Rate direction",
        macro_state["rate_direction"],
        delta=f"{macro_state['rate_level'] - macro_state['prev_rate']:+.2f}%"
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

    preset = st.selectbox("Quick pick (top firms)", PORTFOLIO_TICKERS, index=1, key="single_preset")
    ticker_input = st.text_input(
        "Ticker (yfinance symbol)",
        value=preset,
        help="Any ticker supported by yfinance",
        key="single_ticker",
    ).upper().strip()

    growth = st.slider("Sales growth (%)", min_value=-10.0, max_value=30.0, value=5.0, step=0.5)
    margin = st.slider("Net margin (%)", min_value=-20.0, max_value=50.0, value=15.0, step=0.5)
    cost_equity = st.slider("Cost of equity (%)", min_value=2.0, max_value=20.0, value=8.0, step=0.5)


if ticker_input:
    try:
        fin = load_company_financials(ticker_input)
    except Exception as e:
        st.error(f"Error loading financials for {ticker_input}: {e}")
    else:
        fin_kpis = add_kpis(fin)

        if fin_kpis.empty:
            st.warning("No annual financials found for this ticker. Try another symbol.")
        else:
            info = fin.get("info") or {}
            valuations = valuation_from_info(info, fin_kpis.iloc[-1].get("Revenue", math.nan))

            st.markdown("### Financial KPIs (latest years)")
            st.dataframe(fin_kpis.round(2))

            # robust, safe formatter
            def fmt_number(x, fmt: str, fallback: str = "N/A") -> str:
                try:
                    if x is None:
                        return fallback
                    if isinstance(x, float) and math.isnan(x):
                        return fallback
                    return format(x, fmt)
                except Exception:
                    return fallback

            # valuation metrics
            val_cols = st.columns(4)
            val_cols[0].metric("Market cap", fmt_number(valuations.get("MarketCap"), ",.0f"))
            val_cols[1].metric("Trailing PE", fmt_number(valuations.get("TrailingPE"), ".2f"))
            val_cols[2].metric("PB", fmt_number(valuations.get("PriceToBook"), ".2f"))
            val_cols[3].metric("FCF yield (%)", fmt_number(valuations.get("FCF_YieldPct"), ".2f"))

            # DCF block
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

            st.markdown("### Relative performance vs macro")
            try:
                price_macro = price_vs_macro(ticker_input, cpi)
                if price_macro.empty:
                    st.info("Not enough price history to plot performance vs macro.")
                else:
                    st.line_chart(price_macro)
            except Exception as e:
                st.warning(f"Could not load price vs macro chart: {e}")


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
