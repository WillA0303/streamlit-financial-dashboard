##############################################
# app.py – Macro–Equity Dashboard (Streamlit)
##############################################

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------
# Configuration
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_PATH = PROJECT_ROOT / "data" / "macro"

# Global top 10 by market cap (mapped to yfinance tickers)
PORTFOLIO_TICKERS = [
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
    labels = list(df.index)
    lower_index = {str(idx).lower(): idx for idx in labels}

    # Exact match
    for cand in candidates:
        c = cand.lower()
        if c in lower_index:
            return lower_index[c]

    # Substring match
    for cand in candidates:
        c = cand.lower()
        for idx in labels:
            if c in str(idx).lower():
                return idx

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

@st.cache_data
def load_macro_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CPI, unemployment and base rate from CSVs in data/macro.
    Each CSV must have columns: Date, Value.
    """
    cpi = pd.read_csv(MACRO_PATH / "cpi_uk.csv", parse_dates=["Date"])
    cpi = (
        cpi.rename(columns={"Value": "CPI"})
        .set_index("Date")
        .sort_index()
        .dropna()
    )

    unemp = pd.read_csv(MACRO_PATH / "unemployment_uk.csv", parse_dates=["Date"])
    unemp = (
        unemp.rename(columns={"Value": "Unemployment"})
        .set_index("Date")
        .sort_index()
        .dropna()
    )

    base_rate = pd.read_csv(MACRO_PATH / "base_rate_uk.csv", parse_dates=["Date"])
    base_rate = (
        base_rate.rename(columns={"Value": "BaseRate"})
        .set_index("Date")
        .sort_index()
        .dropna()
    )

    return cpi, unemp, base_rate


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
    unemployment_regime = classify_unemployment(latatest_unemp := latest_unemp)

    summary = f"""
**UK Macro Summary** (sample ending {latest_date})

- CPI level: {latest_cpi:.2f} | YoY: {latest_cpi_yoy:.2f}% | Direction: {cpi_dir} | Regime: {inflation_regime}
- Unemployment: {latest_unemp:.2f}% | YoY: {latest_unemp_yoy:.2f}% | Direction: {unemp_dir} | Regime: {unemployment_regime}
- Bank Rate: {latest_rate:.2f}% | Change vs last: {rate_change:+.2f} | Direction: {rate_dir}
"""
    return summary


def infer_macro_state(cpi_df, unemp_df, rate_df):
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
        "as_of": latest_idx.date(),
        "inflation_regime": classify_inflation(latest_cpi_yoy),
        "unemployment_regime": classify_unemployment(latest_unemp),
        "rate_direction": rate_direction,
    }


# -----------------------
# Equity engine
# -----------------------

@st.cache_data
def load_company_financials(symbol: str, max_years: int = 5) -> pd.DataFrame:
    """
    Load multi-year financials for a company using yfinance.

    Returns DataFrame indexed by Year with:
        Revenue, COGS, OperatingIncome, NetIncome,
        TotalAssets, TotalEquity, TotalDebt, CFO, Capex
    """
    t = yf.Ticker(symbol)

    is_df = t.financials
    bs_df = t.balance_sheet
    cf_df = t.cashflow

    if is_df.empty or bs_df.empty or cf_df.empty:
        raise ValueError(f"Missing financial data for ticker {symbol}")

    revenue_label = _find_row_label(is_df, ["Total Revenue", "Revenue"])
    cogs_label = _find_row_label(is_df, ["Cost Of Revenue", "Cost of Revenue"])
    op_inc_label = _find_row_label(is_df, ["Operating Income"])
    net_inc_label = _find_row_label(is_df, ["Net Income"])

    assets_label = _find_row_label(bs_df, ["Total Assets"])
    equity_label = _find_equity_label(bs_df)

    total_debt_label = _find_row_label(bs_df, ["Total Debt"])
    short_debt_label = _find_row_label(bs_df, ["Short Long Term Debt", "Short Term Debt"])
    long_debt_label = _find_row_label(bs_df, ["Long Term Debt"])

    cfo_label = _find_row_label(
        cf_df,
        ["Total Cash From Operating Activities", "Operating Cash Flow"],
    )
    capex_label = _find_row_label(
        cf_df,
        ["Capital Expenditures", "Capital Expenditure"],
    )

    def get_from(df: pd.DataFrame, label: Optional[str], col) -> float:
        if label is None:
            return math.nan
        if label in df.index and col in df.columns:
            return _to_float(df.loc[label, col])
        return math.nan

    rows: List[Dict[str, Any]] = []

    for col in is_df.columns:
        if hasattr(col, "year"):
            year = int(col.year)
            date_key = col
        else:
            year = int(str(col)[:4])
            date_key = col

        revenue = get_from(is_df, revenue_label, date_key)
        cogs = get_from(is_df, cogs_label, date_key)
        op_inc = get_from(is_df, op_inc_label, date_key)
        net_inc = get_from(is_df, net_inc_label, date_key)

        total_assets = get_from(bs_df, assets_label, date_key)
        total_equity = get_from(bs_df, equity_label, date_key)

        total_debt = get_from(bs_df, total_debt_label, date_key)
        if math.isnan(total_debt):
            short_debt = get_from(bs_df, short_debt_label, date_key)
            long_debt = get_from(bs_df, long_debt_label, date_key)
            if not math.isnan(short_debt) or not math.isnan(long_debt):
                total_debt = 0.0
                if not math.isnan(short_debt):
                    total_debt += short_debt
                if not math.isnan(long_debt):
                    total_debt += long_debt

        cfo = get_from(cf_df, cfo_label, date_key)
        capex = get_from(cf_df, capex_label, date_key)

        rows.append(
            {
                "Year": year,
                "Revenue": revenue,
                "COGS": cogs,
                "OperatingIncome": op_inc,
                "NetIncome": net_inc,
                "TotalAssets": total_assets,
                "TotalEquity": total_equity,
                "TotalDebt": total_debt,
                "CFO": cfo,
                "Capex": capex,
            }
        )

    fin = pd.DataFrame(rows)
    fin = fin.dropna(subset=["Revenue"])
    fin = fin.sort_values("Year").drop_duplicates(subset=["Year"], keep="last")

    if len(fin) > max_years:
        fin = fin.iloc[-max_years:]

    fin = fin.set_index("Year").sort_index()

    return fin


def add_kpis(fin_df: pd.DataFrame) -> pd.DataFrame:
    df = fin_df.copy()

    df["RevenueGrowthPct"] = df["Revenue"].pct_change() * 100
    df["GrossProfit"] = df["Revenue"] - df["COGS"]
    df["GrossMarginPct"] = df["GrossProfit"] / df["Revenue"] * 100
    df["OperatingMarginPct"] = df["OperatingIncome"] / df["Revenue"] * 100
    df["NetMarginPct"] = df["NetIncome"] / df["Revenue"] * 100
    df["ROA_Pct"] = df["NetIncome"] / df["TotalAssets"] * 100
    df["ROE_Pct"] = df["NetIncome"] / df["TotalEquity"] * 100
    df["DebtToEquity"] = df["TotalDebt"] / df["TotalEquity"]
    df["CashConversionPct"] = df["CFO"] / df["NetIncome"] * 100
    df["CapexToRevenuePct"] = df["Capex"] / df["Revenue"] * 100

    return df


def safe_number(x, fmt="{:,.0f}"):
    return "N/A" if pd.isna(x) else fmt.format(x)


def safe_pct(x, fmt="{:.1f}%"):
    return "N/A" if pd.isna(x) else fmt.format(x)


def equity_snapshot_text(ticker: str, fin_kpis: pd.DataFrame) -> str:
    latest = fin_kpis.index.max()
    row = fin_kpis.loc[latest]

    txt = f"""
**{ticker} – Equity Snapshot (Year {latest})**

Scale and growth:
- Revenue: {safe_number(row['Revenue'])}
- Revenue growth: {safe_pct(row['RevenueGrowthPct'])}

Profitability:
- Gross margin: {safe_pct(row['GrossMarginPct'])}
- Operating margin: {safe_pct(row['OperatingMarginPct'])}
- Net margin: {safe_pct(row['NetMarginPct'])}

Returns:
- ROA: {safe_pct(row['ROA_Pct'])}
- ROE: {safe_pct(row['ROE_Pct'])}

Leverage:
- Total Assets: {safe_number(row['TotalAssets'])}
- Total Equity: {safe_number(row['TotalEquity'])}
- Total Debt: {safe_number(row['TotalDebt'])}
- Debt/Equity: {(
    "N/A" if pd.isna(row['DebtToEquity']) else f"{row['DebtToEquity']:.2f}x"
)}

Cash:
- Cash conversion: {safe_pct(row['CashConversionPct'])}
- Capex intensity: {safe_pct(row['CapexToRevenuePct'])}
"""
    return txt


def combined_view_text(ticker: str, fin_kpis: pd.DataFrame, macro_state: Dict[str, Any]) -> str:
    latest = fin_kpis.index.max()
    row = fin_kpis.loc[latest]

    inflation = macro_state["inflation_regime"]
    unemployment = macro_state["unemployment_regime"]
    rate_dir = macro_state["rate_direction"]
    as_of = macro_state["as_of"]

    d_e = row["DebtToEquity"]
    gross_margin = row["GrossMarginPct"]
    rev_growth = row["RevenueGrowthPct"]
    roe = row["ROE_Pct"]

    bullets = []

    if rate_dir == "rising" and not pd.isna(d_e):
        if d_e > 1.0:
            bullets.append(
                f"{ticker} has relatively high leverage ({d_e:.2f}x). Higher rates increase refinancing and interest cost risk."
            )
        elif d_e < 0.3:
            bullets.append(
                f"{ticker} has modest leverage ({d_e:.2f}x) so is less sensitive to higher rates."
            )

    if inflation == "elevated" and not pd.isna(gross_margin):
        if gross_margin < 30:
            bullets.append(
                f"Elevated inflation and thin gross margins (~{gross_margin:.1f}%) leave limited room to absorb input cost shocks."
            )
        elif gross_margin > 50:
            bullets.append(
                f"Strong gross margins (~{gross_margin:.1f}%) give {ticker} pricing power even in an elevated inflation environment."
            )

    if not pd.isna(rev_growth) and not pd.isna(roe):
        if rev_growth > 5 and roe > 15:
            bullets.append(
                f"Despite the macro backdrop (inflation: {inflation}, rates: {rate_dir}), {ticker} still delivers solid fundamentals: {rev_growth:.1f}% revenue growth and ROE of {roe:.1f}%."
            )

    if not bullets:
        bullets.append(
            f"Macro as of {as_of}: inflation '{inflation}', unemployment '{unemployment}', rates '{rate_dir}'. No strong simple macro warning flags in this rule set."
        )

    text = f"**{ticker} – Macro aware view (as of {as_of})**\n\n"
    text += "\n".join(f"- {b}" for b in bullets)
    return text


# -----------------------
# Portfolio helpers
# -----------------------

@st.cache_data
def build_portfolio_table(tickers: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    rows = []
    failures: List[Tuple[str, str]] = []

    for t in tickers:
        t_clean = t.upper().strip()
        try:
            fin = load_company_financials(t_clean)
            kpis = add_kpis(fin)
            latest_year = kpis.index.max()
            r = kpis.loc[latest_year]

            rows.append(
                {
                    "Ticker": t_clean,
                    "Year": latest_year,
                    "Revenue": r["Revenue"],
                    "RevenueGrowthPct": r["RevenueGrowthPct"],
                    "NetMarginPct": r["NetMarginPct"],
                    "ROA_Pct": r["ROA_Pct"],
                    "ROE_Pct": r["ROE_Pct"],
                    "DebtToEquity": r["DebtToEquity"],
                    "CashConversionPct": r["CashConversionPct"],
                    "CapexToRevenuePct": r["CapexToRevenuePct"],
                }
            )
        except Exception as e:
            failures.append((t_clean, str(e)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Ticker").sort_index()

    return df, failures


# -----------------------
# Streamlit layout
# -----------------------

st.set_page_config(
    page_title="Macro–Equity Dashboard",
    layout="wide",
)

st.title("Macro–Equity Dashboard")
st.caption("UK macro context + global megacap fundamentals (live yfinance data).")

# Load macro on app start
try:
    cpi_raw, unemp_raw, rate_raw = load_macro_data()
    cpi, unemp = add_macro_features(cpi_raw, unemp_raw)
    macro_state = infer_macro_state(cpi, unemp, rate_raw)
    macro_summary = macro_summary_text(cpi, unemp, rate_raw)
except Exception as e:
    st.error(f"Error loading macro data: {e}")
    st.stop()

# Sidebar mode selection
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Portfolio (Top 10)", "Single company"],
)

# Macro overview at top
st.subheader("Macro overview")
st.markdown(macro_summary)

macro_cols = st.columns(3)
with macro_cols[0]:
    st.markdown("**CPI (last 24 obs)**")
    st.line_chart(cpi.tail(24)["CPI"])
with macro_cols[1]:
    st.markdown("**Unemployment (all)**")
    st.line_chart(unemp["Unemployment"])
with macro_cols[2]:
    st.markdown("**Bank Rate (all)**")
    st.line_chart(rate_raw["BaseRate"])

st.markdown("---")

# -----------------------
# Mode: Single company
# -----------------------
if mode == "Single company":
    st.subheader("Single company analysis")

    default_ticker = "AAPL"
    ticker_input = st.text_input("Ticker (yfinance symbol)", value=default_ticker).upper().strip()

    if ticker_input:
        try:
            fin = load_company_financials(ticker_input)
            fin_kpis = add_kpis(fin)

            st.markdown("### Financial KPIs (latest years)")
            st.dataframe(fin_kpis.round(2))

            st.markdown("### Equity snapshot")
            st.markdown(equity_snapshot_text(ticker_input, fin_kpis))

            st.markdown("### Macro aware view")
            st.markdown(combined_view_text(ticker_input, fin_kpis, macro_state))

            # Simple charts
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
    st.subheader("Portfolio of top 10 global firms")

    st.markdown(f"Default portfolio tickers: `{', '.join(PORTFOLIO_TICKERS)}`")

    portfolio_df, failures = build_portfolio_table(PORTFOLIO_TICKERS)

    if portfolio_df.empty:
        st.error("Could not build portfolio table (no tickers succeeded).")
    else:
        st.markdown("### Portfolio fundamentals (latest year per company)")
        st.dataframe(portfolio_df.round(2))

        # Charts
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
            st.markdown("### Tickers with errors")
            for t, msg in failures:
                st.write(f"- {t}: {msg}")

        st.markdown("---")
        st.markdown("### Macro aware notes per company")
        for t in portfolio_df.index:
            try:
                fin = load_company_financials(t)
                kpis = add_kpis(fin)
                note = combined_view_text(t, kpis, macro_state)
                st.markdown(note)
                st.markdown("---")
            except Exception as e:
                st.write(f"Error for {t}: {e}")
