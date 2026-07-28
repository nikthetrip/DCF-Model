from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_provider import CompanyData, fetch_fmp, fetch_yfinance
from export_model import build_excel_export
from valuation import (
    ValuationInputs,
    calculate_quality_metrics,
    owner_earnings_bridge,
    run_dcf,
    scenario_values,
    sensitivity_table,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "assets" / "Buffett_Munger_Owner_Earnings_Model.xlsx"
TICKERS_PATH = BASE_DIR / "tickers.csv"

st.set_page_config(
    page_title="Owner Earnings Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-safe styling: do not force light metric cards inside dark mode.
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.28);
        padding: 14px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    .small-note {opacity: .72; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_KEYS = {
    "price", "shares", "revenue", "ebit", "interest_expense", "pretax_income", "tax_expense",
    "reported_net_income", "normalized_net_income", "da", "provider_other_non_cash", "other_non_cash",
    "sbc", "deduct_sbc", "total_capex", "maintenance_method", "maintenance_pct", "maintenance_capex_manual",
    "wc_method", "working_capital", "ocf", "fcf", "cash", "debt", "assets", "equity", "invested_capital",
    "goodwill_intangibles", "excess_cash", "non_operating_assets", "other_claims", "other_adjustments",
    "normalized_tax_rate", "start_growth", "year_10_growth", "terminal_growth", "discount_rate",
    "history_editor_widget", "_last_wc_method",
    "margin_target", "minimum_oe_yield", "bear_growth", "bear_discount", "bear_terminal", "bear_probability",
    "base_probability", "bull_growth", "bull_discount", "bull_terminal", "bull_probability", "history_editor",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(symbol: str, provider: str, api_key: str | None) -> CompanyData:
    if provider == "Financial Modeling Prep":
        return fetch_fmp(symbol, api_key)
    return fetch_yfinance(symbol)


def clear_model_state() -> None:
    for key in MODEL_KEYS:
        st.session_state.pop(key, None)


def value_or_zero(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def value_or_none(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def money(value: float | None, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{value:,.2f}"


def money_mm(value: float | None, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{value:,.1f}m"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}"


def multiple(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1f}x"


def set_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def number_input(label: str, key: str, default: float, help_text: str, step: float = 1.0, **kwargs) -> float:
    set_default(key, float(default))
    return float(st.number_input(label, key=key, step=step, help=help_text, **kwargs))


def inferred_tax_rate(ltm: dict[str, float | None]) -> float:
    pretax = value_or_none(ltm.get("pretax_income"))
    tax = value_or_none(ltm.get("tax_expense"))
    if pretax and pretax > 0 and tax is not None:
        rate = tax / pretax
        if 0 <= rate <= 0.50:
            return float(rate)
    return 0.21


def wc_cash_requirement(raw_change: float | None) -> float:
    """Heuristic only: assumes a negative provider change-in-WC value is a cash use."""
    if raw_change is None or pd.isna(raw_change):
        return 0.0
    return max(-float(raw_change), 0.0)


def average_wc_requirement(frame: pd.DataFrame, years: int) -> float:
    if frame.empty or "change_in_working_capital" not in frame.columns:
        return 0.0
    vals = pd.to_numeric(frame.tail(years)["change_in_working_capital"], errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(np.mean([wc_cash_requirement(v) for v in vals]))


def data_quality_table(company: CompanyData) -> pd.DataFrame:
    fields = [
        ("Share price", company.price, "Market data"),
        ("Revenue", company.ltm.get("revenue"), "Income statement"),
        ("EBIT / operating income", company.ltm.get("ebit"), "Income statement"),
        ("Net income", company.ltm.get("net_income"), "Income statement"),
        ("Diluted shares", company.ltm.get("diluted_shares"), "Income statement"),
        ("D&A", company.ltm.get("depreciation_amortization"), "Cash flow statement"),
        ("Total capex", company.ltm.get("capital_expenditure"), "Cash flow statement"),
        ("Operating cash flow", company.ltm.get("operating_cash_flow"), "Cash flow statement"),
        ("Cash", company.ltm.get("cash"), "Balance sheet"),
        ("Total debt", company.ltm.get("total_debt"), "Balance sheet"),
        ("Invested capital", company.ltm.get("invested_capital"), "Balance sheet / derived"),
    ]
    return pd.DataFrame(
        [{"Field": label, "Status": "Available" if value is not None and not pd.isna(value) else "Missing", "Source area": area} for label, value, area in fields]
    )


def quality_completeness(company: CompanyData) -> float:
    table = data_quality_table(company)
    return float((table["Status"] == "Available").mean()) if not table.empty else 0.0


# ---------------- Sidebar / load data ----------------
tickers = pd.read_csv(TICKERS_PATH)
ticker_labels = [f"{row.symbol} — {row.name}" for row in tickers.itertuples(index=False)]

with st.sidebar:
    st.title("Owner Earnings Model")
    st.caption("Buffett–Munger inspired valuation framework")
    provider = st.selectbox(
        "Data provider",
        ["Yahoo Finance (no key)", "Financial Modeling Prep"],
        help="Yahoo Finance is convenient but can have missing or inconsistent classifications. FMP generally provides cleaner standardized statements and requires an API key.",
    )
    api_key = None
    if provider == "Financial Modeling Prep":
        default_secret = None
        try:
            default_secret = st.secrets.get("FMP_API_KEY")
        except Exception:
            default_secret = None
        api_key = st.text_input(
            "FMP API key",
            value="" if default_secret is None else str(default_secret),
            type="password",
            help="For deployment, store FMP_API_KEY in Streamlit secrets rather than in GitHub.",
        )

    default_label = "FI — Fiserv Inc."
    selected_label = st.selectbox(
        "Ticker",
        ticker_labels,
        index=ticker_labels.index(default_label) if default_label in ticker_labels else 0,
        help="Choose a ticker or enter another symbol below.",
    )
    custom_ticker = st.text_input(
        "Custom ticker (optional)",
        placeholder="Example: SPOT or LVMUY",
        help="When provided, this symbol overrides the dropdown selection.",
    ).strip().upper()
    selected_symbol = custom_ticker or selected_label.split(" — ")[0]
    fetch_clicked = st.button("Load financial data", type="primary", use_container_width=True)
    if st.session_state.get("loaded_symbol"):
        st.caption(f"Loaded: {st.session_state.loaded_symbol}")
    st.divider()
    st.caption("Research tool only. Reconcile key figures to company filings before relying on the output.")

if fetch_clicked or "company_data" not in st.session_state:
    try:
        with st.spinner(f"Loading {selected_symbol} financial data..."):
            loaded = load_company(selected_symbol, provider, api_key)
            clear_model_state()
            st.session_state.company_data = loaded
            st.session_state.loaded_symbol = selected_symbol
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")
        st.info("Verify the ticker/provider and, when using FMP, the API key and plan access.")
        st.stop()

company: CompanyData = st.session_state.company_data
ltm = company.ltm
historical_raw = company.historical.copy()
currency = company.currency or "USD"

st.title("Owner Earnings Valuation")
st.caption(
    f"{company.name} ({company.symbol}) · Source: {company.source} · Statement values are shown in {currency} millions unless stated otherwise."
)
for warning in company.warnings:
    st.warning(warning)

# Fetched defaults
price_default = value_or_zero(company.price)
shares_default = value_or_zero(ltm.get("diluted_shares"))
revenue_default = value_or_zero(ltm.get("revenue"))
ebit_default = value_or_zero(ltm.get("ebit"))
interest_default = value_or_zero(ltm.get("interest_expense"))
pretax_default = value_or_zero(ltm.get("pretax_income"))
tax_expense_default = value_or_zero(ltm.get("tax_expense"))
net_income_default = value_or_zero(ltm.get("net_income"))
da_default = value_or_zero(ltm.get("depreciation_amortization"))
provider_other_non_cash_default = value_or_zero(ltm.get("other_non_cash"))
sbc_default = value_or_zero(ltm.get("stock_based_compensation"))
capex_default = value_or_zero(ltm.get("capital_expenditure"))
ocf_default = value_or_zero(ltm.get("operating_cash_flow"))
fcf_default = value_or_zero(ltm.get("free_cash_flow"))
cash_default = value_or_zero(ltm.get("cash"))
debt_default = value_or_zero(ltm.get("total_debt"))
assets_default = value_or_zero(ltm.get("total_assets"))
equity_default = value_or_zero(ltm.get("equity"))
invested_capital_default = value_or_zero(ltm.get("invested_capital"))
goodwill_default = value_or_zero(ltm.get("goodwill_intangibles"))
ltm_wc_heuristic = wc_cash_requirement(value_or_none(ltm.get("change_in_working_capital")))
wc_3y = average_wc_requirement(historical_raw, 3)
wc_5y = average_wc_requirement(historical_raw, 5)

(
    overview_tab,
    financial_tab,
    normalization_tab,
    assumptions_tab,
    quality_tab,
    valuation_tab,
    history_tab,
    method_tab,
) = st.tabs(["Overview", "Financial Data", "Normalization", "Assumptions", "Quality", "Valuation", "Historical Data", "Methodology"])

# ---------------- Financial data: facts ----------------
with financial_tab:
    st.subheader("Fetched financial data")
    st.caption("These are provider values, not model judgments. They remain editable so you can reconcile them to the latest filing.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        price = number_input("Current share price", "price", price_default, "Current market price per share.", step=0.1)
        shares = number_input("Diluted shares outstanding (mm)", "shares", shares_default, "Current or representative diluted share count in millions.", step=1.0)
        revenue = number_input("Revenue ($mm)", "revenue", revenue_default, "LTM reported revenue.", step=10.0)
        ebit = number_input("EBIT / operating income ($mm)", "ebit", ebit_default, "LTM operating income / EBIT.", step=10.0)
    with col2:
        interest_expense = number_input("Interest expense ($mm)", "interest_expense", interest_default, "LTM interest expense. Used for interest coverage.", step=10.0)
        pretax_income = number_input("Pretax income ($mm)", "pretax_income", pretax_default, "LTM income before tax.", step=10.0)
        tax_expense = number_input("Income tax expense ($mm)", "tax_expense", tax_expense_default, "LTM reported income tax expense.", step=10.0)
        reported_net_income = number_input("Reported LTM net income ($mm)", "reported_net_income", net_income_default, "Reported LTM net income attributable to common shareholders.", step=10.0)
    with col3:
        total_capex = number_input("Total capital expenditure ($mm)", "total_capex", capex_default, "Total LTM capital expenditure. This is not automatically the same as maintenance capex.", step=10.0)
        ocf = number_input("Operating cash flow ($mm)", "ocf", ocf_default, "LTM cash from operations.", step=10.0)
        fcf = number_input("Free cash flow ($mm)", "fcf", fcf_default, "Provider FCF or operating cash flow less total capex.", step=10.0)
        cash = number_input("Cash & short-term investments ($mm)", "cash", cash_default, "Latest cash and short-term investments.", step=10.0)
    with col4:
        debt = number_input("Total debt ($mm)", "debt", debt_default, "Latest interest-bearing debt.", step=10.0)
        assets = number_input("Total assets ($mm)", "assets", assets_default, "Latest total assets.", step=10.0)
        equity = number_input("Shareholders' equity ($mm)", "equity", equity_default, "Latest common shareholders' equity.", step=10.0)
        invested_capital = number_input("Invested capital ($mm)", "invested_capital", invested_capital_default, "Provider invested capital or debt + equity − cash when derived.", step=10.0)
        goodwill_intangibles = number_input("Goodwill & intangibles ($mm)", "goodwill_intangibles", goodwill_default, "Latest goodwill and acquired intangible assets.", step=10.0)

    st.subheader("Data quality")
    completeness = quality_completeness(company)
    q1, q2, q3 = st.columns(3)
    q1.metric("Provider completeness", pct(completeness), help="Share of core fields returned by the selected provider. Completeness does not guarantee correctness.")
    q2.metric("Historical annual periods", str(len(historical_raw)))
    q3.metric("Provider", company.source)
    quality_check = data_quality_table(company)
    st.dataframe(quality_check, hide_index=True, use_container_width=True)
    st.info("Maintenance capex, normalized working-capital needs and recurring non-cash adjustments are investor judgments. They are deliberately handled in the Normalization tab rather than trusted mechanically from the provider.")

# ---------------- Normalization: judgments ----------------
with normalization_tab:
    st.subheader("Normalize earning power")
    st.caption("Keep reported facts separate from investor judgments. The defaults are intentionally conservative where provider classifications are ambiguous.")

    left, middle, right = st.columns(3)
    with left:
        st.metric("Reported LTM net income", money_mm(reported_net_income, currency))
        normalized_net_income = number_input(
            "Normalized net income ($mm)", "normalized_net_income", reported_net_income,
            "Start from reported net income, then remove one-time gains/losses, unusual tax items, restructuring, impairments or other non-recurring effects.", step=10.0,
        )
        da = number_input("Depreciation & amortization ($mm)", "da", da_default, "Recurring D&A added back in the owner-earnings bridge.", step=10.0)
        sbc = number_input("Stock-based compensation ($mm)", "sbc", sbc_default, "Employee equity compensation. Deducting it treats dilution/equity compensation as an economic cost.", step=10.0)
        set_default("deduct_sbc", True)
        deduct_sbc = st.toggle("Deduct stock-based compensation", key="deduct_sbc", help="Enabled by default as a conservative treatment.")

    with middle:
        st.metric("Provider 'other non-cash items'", money_mm(provider_other_non_cash_default, currency), help="Reference only. Provider classifications can mix recurring and non-recurring items.")
        other_non_cash = number_input(
            "Recurring non-cash adjustments to add back ($mm)", "other_non_cash", 0.0,
            "Manual amount only. Do not automatically copy the provider's 'other non-cash items' without reviewing the filing.", step=10.0,
        )
        set_default("maintenance_method", "Percentage of total capex")
        maintenance_method = st.radio(
            "Maintenance capex method",
            ["Percentage of total capex", "Manual amount"],
            key="maintenance_method",
            help="Maintenance capex is spending required to preserve current earning capacity. Growth capex should not be deducted from current earning power unless it is necessary to maintain the business.",
        )
        if maintenance_method == "Percentage of total capex":
            set_default("maintenance_pct", 0.70)
            maintenance_pct = st.slider(
                "Maintenance capex as % of total capex", min_value=0.0, max_value=1.50, step=0.05,
                key="maintenance_pct", help="Judgmental estimate. Compare with depreciation, asset age, capacity additions and management disclosures.",
            )
            maintenance_capex = total_capex * maintenance_pct
            st.caption(f"Calculated maintenance capex: {money_mm(maintenance_capex, currency)} from total capex of {money_mm(total_capex, currency)}.")
        else:
            maintenance_capex = number_input(
                "Manual maintenance capex ($mm)", "maintenance_capex_manual", total_capex * 0.70,
                "Direct estimate of the capital spending required to maintain current earning capacity.", step=10.0,
            )

    with right:
        set_default("wc_method", "Manual")
        wc_method = st.selectbox(
            "Recurring working-capital method",
            ["Manual", "LTM heuristic", "3-year average heuristic", "5-year average heuristic"],
            key="wc_method",
            help="Working-capital cash needs are noisy. The heuristic assumes a negative provider change-in-working-capital figure is a use of cash; verify the provider sign convention and the filing.",
        )
        wc_suggestion = {
            "Manual": 0.0,
            "LTM heuristic": ltm_wc_heuristic,
            "3-year average heuristic": wc_3y,
            "5-year average heuristic": wc_5y,
        }[wc_method]
        if wc_method != "Manual":
            st.caption(f"Heuristic suggestion: {money_mm(wc_suggestion, currency)}. You can still override it below.")
        set_default("working_capital", wc_suggestion)
        if wc_method != "Manual" and st.session_state.get("_last_wc_method") != wc_method:
            st.session_state.working_capital = wc_suggestion
        st.session_state["_last_wc_method"] = wc_method
        working_capital = float(st.number_input(
            "Recurring working-capital investment ($mm)", key="working_capital", step=10.0,
            help="Positive amount reduces owner earnings. Use a normalized recurring cash requirement rather than a noisy single-period movement.",
        ))

        st.metric("Latest cash reported", money_mm(cash, currency))
        excess_cash = number_input("Excess cash added to equity value ($mm)", "excess_cash", 0.0, "Only cash demonstrably not required for normal operations. Default is zero rather than an automatic formula.", step=10.0)
        non_operating_assets = number_input("Non-operating investments / assets ($mm)", "non_operating_assets", 0.0, "Investments or other separable non-operating assets to add to equity value.", step=10.0)
        other_claims = number_input("Preferred / minority / pension / other claims ($mm)", "other_claims", 0.0, "Claims senior to common equity that should be deducted from equity value.", step=10.0)
        other_adjustments = number_input("Other net equity adjustments ($mm)", "other_adjustments", 0.0, "Any additional net adjustment not captured above. Positive adds value; negative subtracts value.", step=10.0)

# ---------------- Assumptions ----------------
with assumptions_tab:
    st.subheader("Valuation assumptions")
    st.caption("These are investor assumptions, not facts supplied by the company.")
    a, b, c, d = st.columns(4)
    with a:
        start_growth = number_input("Starting owner earnings growth", "start_growth", 0.08, "Growth applied in forecast year 1.", step=0.005, min_value=-0.50, max_value=0.50, format="%.3f")
        year_10_growth = number_input("Year 10 growth", "year_10_growth", 0.04, "Growth fades linearly from year 1 to this rate in year 10.", step=0.005, min_value=-0.20, max_value=0.20, format="%.3f")
    with b:
        terminal_growth = number_input("Terminal growth", "terminal_growth", 0.025, "Perpetual growth after year 10. It must remain below the discount rate.", step=0.0025, min_value=-0.05, max_value=0.08, format="%.4f")
        discount_rate = number_input("Required return / discount rate", "discount_rate", 0.09, "Opportunity cost adjusted for business risk and forecast uncertainty.", step=0.005, min_value=0.01, max_value=0.30, format="%.3f")
    with c:
        margin_target = number_input("Margin of safety target", "margin_target", 0.25, "Target discount from intrinsic value to purchase price.", step=0.05, min_value=0.0, max_value=0.80, format="%.2f")
        minimum_oe_yield = number_input("Minimum owner earnings yield", "minimum_oe_yield", 0.06, "Alternative price discipline: normalized owner earnings per share divided by price.", step=0.005, min_value=0.01, max_value=0.30, format="%.3f")
    with d:
        normalized_tax_rate = number_input("Normalized tax rate", "normalized_tax_rate", inferred_tax_rate(ltm), "Used for NOPAT and ROIC. The default is the LTM effective rate when sensible, otherwise 21%.", step=0.01, min_value=0.0, max_value=0.60, format="%.3f")
        st.info("Terminal value often represents a large share of a DCF. Use the sensitivity table and keep terminal growth conservative.")

    st.subheader("Scenario assumptions")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Bear case**")
        bear_growth = number_input("Starting growth", "bear_growth", 0.03, "Bear-case starting growth.", step=0.005, min_value=-0.50, max_value=0.50, format="%.3f")
        bear_discount = number_input("Discount rate", "bear_discount", 0.11, "Bear-case required return.", step=0.005, min_value=0.01, max_value=0.30, format="%.3f")
        bear_terminal = number_input("Terminal growth", "bear_terminal", 0.015, "Bear-case terminal growth.", step=0.0025, min_value=-0.05, max_value=0.08, format="%.4f")
        bear_probability = number_input("Probability", "bear_probability", 0.25, "Bear-case probability.", step=0.05, min_value=0.0, max_value=1.0, format="%.2f")
    with s2:
        st.markdown("**Base case**")
        st.write("Uses the base growth, discount rate and terminal growth above.")
        base_probability = number_input("Base probability", "base_probability", 0.50, "Base-case probability.", step=0.05, min_value=0.0, max_value=1.0, format="%.2f")
    with s3:
        st.markdown("**Bull case**")
        bull_growth = number_input("Starting growth", "bull_growth", 0.11, "Bull-case starting growth.", step=0.005, min_value=-0.50, max_value=0.50, format="%.3f")
        bull_discount = number_input("Discount rate", "bull_discount", 0.08, "Bull-case required return.", step=0.005, min_value=0.01, max_value=0.30, format="%.3f")
        bull_terminal = number_input("Terminal growth", "bull_terminal", 0.03, "Bull-case terminal growth.", step=0.0025, min_value=-0.05, max_value=0.08, format="%.4f")
        bull_probability = number_input("Probability", "bull_probability", 0.25, "Bull-case probability.", step=0.05, min_value=0.0, max_value=1.0, format="%.2f")

# Editable historical table before quality metrics are calculated.
with history_tab:
    st.subheader("Historical annual data")
    st.caption("Provider values are editable. Reconcile unusual or missing figures to annual reports. Quality metrics use the edited table below.")
    if historical_raw.empty:
        st.info("No historical annual data were returned.")
        historical = pd.DataFrame()
    else:
        set_default("history_editor", historical_raw)
        historical = st.data_editor(
            st.session_state.history_editor,
            key="history_editor_widget",
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
        )
        st.session_state.history_editor = historical

# ---------------- Calculate model ----------------
inputs = ValuationInputs(
    price=price,
    diluted_shares_mm=shares,
    net_income_mm=normalized_net_income,
    depreciation_amortization_mm=da,
    other_non_cash_mm=other_non_cash,
    stock_based_compensation_mm=sbc,
    maintenance_capex_mm=maintenance_capex,
    working_capital_investment_mm=working_capital,
    excess_cash_mm=excess_cash,
    non_operating_assets_mm=non_operating_assets,
    other_claims_mm=other_claims,
    other_equity_adjustments_mm=other_adjustments,
    deduct_sbc=deduct_sbc,
    start_growth=start_growth,
    year_10_growth=year_10_growth,
    terminal_growth=terminal_growth,
    discount_rate=discount_rate,
    margin_of_safety_target=margin_target,
    minimum_owner_earnings_yield=minimum_oe_yield,
)

model_error: str | None = None
try:
    result = run_dcf(inputs)
except ValueError as exc:
    model_error = str(exc)
    result = None

scenario_prob_total = bear_probability + base_probability + bull_probability
scenarios = {
    "Bear": {"start_growth": bear_growth, "discount_rate": bear_discount, "terminal_growth": bear_terminal, "probability": bear_probability},
    "Base": {"start_growth": start_growth, "discount_rate": discount_rate, "terminal_growth": terminal_growth, "probability": base_probability},
    "Bull": {"start_growth": bull_growth, "discount_rate": bull_discount, "terminal_growth": bull_terminal, "probability": bull_probability},
}
scenario_df = pd.DataFrame()
weighted_value = np.nan
if result is not None:
    try:
        scenario_df = scenario_values(inputs, scenarios)
        if scenario_prob_total > 0:
            weighted_value = scenario_df["Weighted value"].sum() / scenario_prob_total
    except ValueError:
        scenario_df = pd.DataFrame()

quality_df = calculate_quality_metrics(historical, tax_rate=normalized_tax_rate) if not historical.empty else pd.DataFrame()

# ---------------- Overview ----------------
with overview_tab:
    if model_error:
        st.error(model_error)
    else:
        cols = st.columns(6)
        cols[0].metric("Current price", money(price, currency))
        cols[1].metric("Owner earnings / share", money(result["owner_earnings_per_share"], currency))
        cols[2].metric("No-growth value", money(result["no_growth_value_per_share"], currency))
        cols[3].metric("DCF intrinsic value", money(result["intrinsic_value_per_share"], currency))
        cols[4].metric("Margin of safety", pct(result["margin_of_safety"]))
        cols[5].metric("Data completeness", pct(quality_completeness(company)))

        left, right = st.columns([1.15, 0.85])
        with left:
            fig = px.line(result["forecast"], x="Year", y="Owner earnings ($mm)", markers=True, title="Base-case owner earnings forecast")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), yaxis_title=f"{currency} millions")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Valuation checks")
            checks = pd.DataFrame(
                {
                    "Measure": ["Base DCF", "Probability-weighted", "Target buy price", "Yield-based buy price"],
                    "Value per share": [result["intrinsic_value_per_share"], weighted_value, result["target_buy_price"], result["yield_based_buy_price"]],
                }
            )
            st.dataframe(checks.style.format({"Value per share": "{:,.2f}"}, na_rep="—"), hide_index=True, use_container_width=True)
            st.caption(f"Current owner earnings yield: {pct(result['owner_earnings_yield'])} · Price / owner earnings: {multiple(result['price_to_owner_earnings'])}")
            st.caption(f"Terminal value share of operating DCF: {pct(result['terminal_value_share'])}")

        st.subheader("Model status")
        status_cols = st.columns(4)
        status_cols[0].metric("Normalized owner earnings", money_mm(result["owner_earnings_mm"], currency))
        status_cols[1].metric("Maintenance capex", money_mm(maintenance_capex, currency))
        status_cols[2].metric("Recurring WC investment", money_mm(working_capital, currency))
        status_cols[3].metric("Scenario probabilities", pct(scenario_prob_total), help="The weighted value is normalized by the total probability if probabilities do not sum to 100%.")
        if abs(scenario_prob_total - 1.0) > 1e-6:
            st.warning("Scenario probabilities do not sum to 100%. The displayed probability-weighted value is normalized by the entered total.")

        st.subheader("Downloads")
        financial_snapshot = {
            "price": price, "shares": shares, "revenue": revenue, "ebit": ebit, "interest_expense": interest_expense,
            "pretax_income": pretax_income, "tax_expense": tax_expense, "reported_net_income": reported_net_income,
            "total_capex": total_capex, "operating_cash_flow": ocf, "free_cash_flow": fcf, "cash": cash, "debt": debt,
            "assets": assets, "equity": equity, "invested_capital": invested_capital, "goodwill_intangibles": goodwill_intangibles,
        }
        normalization_snapshot = {
            "normalized_net_income": normalized_net_income, "depreciation_amortization": da, "stock_based_compensation": sbc,
            "deduct_sbc": deduct_sbc, "provider_other_non_cash_reference": provider_other_non_cash_default,
            "recurring_other_non_cash_addback": other_non_cash, "maintenance_capex": maintenance_capex,
            "working_capital_investment": working_capital, "excess_cash": excess_cash, "non_operating_assets": non_operating_assets,
            "other_claims": other_claims, "other_adjustments": other_adjustments, "normalized_tax_rate": normalized_tax_rate,
        }
        export_bytes = build_excel_export(
            {"name": company.name, "symbol": company.symbol, "price": price, "source": company.source},
            historical,
            result,
            scenario_df,
            financial_snapshot=financial_snapshot,
            normalization_snapshot=normalization_snapshot,
        )
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Download populated analysis (.xlsx)", export_bytes, f"{company.symbol}_owner_earnings_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        dl2.download_button("Download original model template", TEMPLATE_PATH.read_bytes(), TEMPLATE_PATH.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        json_payload = {k: v for k, v in result.items() if k != "forecast"}
        json_payload["forecast"] = result["forecast"].to_dict(orient="records")
        json_payload["financial_snapshot"] = financial_snapshot
        json_payload["normalization_snapshot"] = normalization_snapshot
        dl3.download_button("Download model data (.json)", json.dumps(json_payload, indent=2, default=str), f"{company.symbol}_valuation.json", "application/json", use_container_width=True)

# ---------------- Quality ----------------
with quality_tab:
    st.subheader("Business quality")
    if quality_df.empty:
        st.info("Historical statements are not available for this ticker.")
    else:
        latest = quality_df.iloc[-1]
        metrics = st.columns(6)
        metrics[0].metric("Latest ROIC", pct(latest.get("roic")))
        metrics[1].metric("EBIT margin", pct(latest.get("ebit_margin")))
        metrics[2].metric("FCF margin", pct(latest.get("fcf_margin")))
        metrics[3].metric("Interest coverage", multiple(latest.get("interest_coverage")))
        metrics[4].metric("OCF / net income", multiple(latest.get("ocf_conversion")))
        metrics[5].metric("Share-count growth", pct(latest.get("share_count_growth")))

        display_cols = [
            "date", "revenue_growth", "ebit_margin", "net_margin", "fcf_margin", "roic", "roa", "roe",
            "interest_coverage", "ocf_conversion", "debt_to_fcf", "share_count_growth", "goodwill_intangibles_to_assets",
        ]
        shown = quality_df[[c for c in display_cols if c in quality_df.columns]].copy()
        st.dataframe(
            shown.style.format(
                {
                    "revenue_growth": "{:.1%}", "ebit_margin": "{:.1%}", "net_margin": "{:.1%}", "fcf_margin": "{:.1%}",
                    "roic": "{:.1%}", "roa": "{:.1%}", "roe": "{:.1%}", "interest_coverage": "{:.1f}x",
                    "ocf_conversion": "{:.1f}x", "debt_to_fcf": "{:.1f}x", "share_count_growth": "{:.1%}",
                    "goodwill_intangibles_to_assets": "{:.1%}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart_cols = [c for c in ["roic", "ebit_margin", "fcf_margin", "net_margin"] if c in quality_df.columns]
        chart_data = quality_df[["date"] + chart_cols].melt("date", var_name="Metric", value_name="Value")
        fig = px.line(chart_data, x="date", y="Value", color="Metric", markers=True, title="Return and margin trends")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Quality metrics are diagnostics, not automatic buy/sell rules. ROIC definitions vary across data providers; verify invested capital when the result looks unusual.")

# ---------------- Valuation ----------------
with valuation_tab:
    if model_error:
        st.error(model_error)
    else:
        st.subheader("Owner earnings bridge")
        bridge = owner_earnings_bridge(inputs)
        st.dataframe(
            bridge.style.format({"Amount ($mm)": "{:,.1f}", "Running owner earnings ($mm)": "{:,.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("DCF forecast")
        st.dataframe(
            result["forecast"].style.format({"Growth rate": "{:.1%}", "Owner earnings ($mm)": "{:,.1f}", "Present value ($mm)": "{:,.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Scenario analysis")
        if scenario_df.empty:
            st.warning("One or more scenario discount rates are not greater than terminal growth. Fix the scenario assumptions.")
        else:
            st.dataframe(
                scenario_df.style.format({"Intrinsic value per share": "{:,.2f}", "Probability": "{:.0%}", "Weighted value": "{:,.2f}", "Margin of safety": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.metric("Probability-weighted intrinsic value", money(weighted_value, currency))

        st.subheader("DCF sensitivity matrix")
        discount_grid = np.array([discount_rate - 0.02, discount_rate - 0.01, discount_rate, discount_rate + 0.01, discount_rate + 0.02])
        discount_grid = np.clip(discount_grid, 0.01, 0.30)
        terminal_grid = np.array([terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01])
        terminal_grid = np.clip(terminal_grid, -0.05, 0.08)
        sens = sensitivity_table(inputs, discount_grid, terminal_grid)
        sens.index = [f"{x:.1%}" for x in sens.index]
        st.dataframe(sens.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
        st.caption("Rows = terminal growth. Columns = required return / discount rate. Cells where terminal growth is not below the discount rate are intentionally blank.")

# Additional historical chart after editor.
with history_tab:
    if not historical.empty and {"date", "revenue", "net_income", "operating_cash_flow"}.issubset(historical.columns):
        chart = historical[["date", "revenue", "net_income", "operating_cash_flow"]].melt("date", var_name="Metric", value_name="Value ($mm)")
        fig = px.bar(chart, x="date", y="Value ($mm)", color="Metric", barmode="group", title="Historical financial performance")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Methodology ----------------
with method_tab:
    st.subheader("Methodology")
    st.markdown(
        """
        **Owner earnings** estimate cash earning power available to common owners without impairing the company's current competitive position:

        `Normalized net income + D&A + reviewed recurring non-cash adjustments − SBC (if selected) − maintenance capex − recurring working-capital investment`

        **No-growth earning power value** capitalizes normalized owner earnings as a perpetuity:

        `Owner earnings per share ÷ required return`

        **DCF intrinsic value** discounts ten years of forecast owner earnings plus a terminal value. Growth fades linearly from the starting growth rate to the year-10 rate.

        Because the model starts from **net income**, interest expense is already reflected in owner earnings. Therefore total debt is used as a quality/risk diagnostic and is **not automatically subtracted again** from equity value. Separate non-operating assets and senior claims can be entered explicitly in Normalization.

        **Required return** is an opportunity-cost assumption. It should reflect alternatives available to the investor, business quality, leverage, cyclicality and forecast uncertainty.

        **Maintenance capex** and **recurring working-capital needs** are judgmental. Provider cash-flow classifications are useful starting points, not authoritative normalized values.
        """
    )
    st.warning("This application automates arithmetic and data collection, not investment judgment. Verify filings, footnotes, acquisition accounting, dilution, debt terms, pension/minority claims and capital-allocation history.")
