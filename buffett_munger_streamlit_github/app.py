from __future__ import annotations

import json
from dataclasses import asdict
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
    run_dcf,
    scenario_values,
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

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1450px;}
    div[data-testid="stMetric"] {background: #F7F9FC; border: 1px solid #E1E7EF; padding: 14px; border-radius: 10px;}
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    .small-note {color: #667085; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(symbol: str, provider: str, api_key: str | None) -> CompanyData:
    if provider == "Financial Modeling Prep":
        return fetch_fmp(symbol, api_key)
    return fetch_yfinance(symbol)


def value_or_zero(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def money(value: float | None, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "—"
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{value:,.2f}"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}"


def num_input(label: str, value: float, help_text: str, key: str, step: float = 1.0) -> float:
    return float(st.number_input(label, value=float(value), step=step, help=help_text, key=key))


tickers = pd.read_csv(TICKERS_PATH)
ticker_labels = [f"{row.symbol} — {row.name}" for row in tickers.itertuples(index=False)]

with st.sidebar:
    st.title("Owner Earnings Model")
    st.caption("Buffett–Munger inspired valuation framework")
    provider = st.selectbox(
        "Data provider",
        ["Yahoo Finance (no key)", "Financial Modeling Prep"],
        help="Yahoo Finance is convenient but can have missing or inconsistent fundamentals. FMP generally provides cleaner standardized statements and requires an API key.",
    )
    api_key = None
    if provider == "Financial Modeling Prep":
        api_key = st.text_input(
            "FMP API key",
            type="password",
            help="Stored only in the current Streamlit session. For deployment, add FMP_API_KEY to Streamlit secrets.",
        )

    selected_label = st.selectbox(
        "Ticker",
        ticker_labels,
        index=ticker_labels.index("FI — Fiserv Inc."),
        help="Choose a common ticker from the list or enter another symbol below.",
    )
    custom_ticker = st.text_input(
        "Custom ticker (optional)",
        placeholder="Example: SPOT or LVMUY",
        help="When provided, this symbol overrides the dropdown selection.",
    ).strip().upper()
    selected_symbol = custom_ticker or selected_label.split(" — ")[0]
    fetch_clicked = st.button("Load financial data", type="primary", use_container_width=True)
    st.divider()
    st.caption("The model is intended for research and education. Verify every input against company filings.")

if fetch_clicked or "company_data" not in st.session_state:
    try:
        with st.spinner(f"Loading {selected_symbol} financial data..."):
            st.session_state.company_data = load_company(selected_symbol, provider, api_key)
            st.session_state.loaded_symbol = selected_symbol
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")
        st.info("Try Yahoo Finance, verify the ticker, or provide a valid FMP API key.")
        st.stop()

company: CompanyData = st.session_state.company_data
ltm = company.ltm
historical = company.historical.copy()

st.title("Owner Earnings Valuation")
st.caption(
    f"{company.name} ({company.symbol}) · Data source: {company.source} · All statement values are shown in $ millions unless stated otherwise."
)
for warning in company.warnings:
    st.warning(warning)

# Defaults fetched from provider; every value remains editable.
price_default = value_or_zero(company.price)
shares_default = value_or_zero(ltm.get("diluted_shares"))
net_income_default = value_or_zero(ltm.get("net_income"))
da_default = value_or_zero(ltm.get("depreciation_amortization"))
other_non_cash_default = value_or_zero(ltm.get("other_non_cash"))
sbc_default = value_or_zero(ltm.get("stock_based_compensation"))
capex_default = value_or_zero(ltm.get("capital_expenditure"))
wc_raw = value_or_zero(ltm.get("change_in_working_capital"))
working_capital_default = max(-wc_raw, 0.0)
cash_default = value_or_zero(ltm.get("cash"))
debt_default = value_or_zero(ltm.get("total_debt"))

summary_tab, inputs_tab, quality_tab, valuation_tab, history_tab, method_tab = st.tabs(
    ["Overview", "Inputs", "Quality", "Valuation", "Historical Data", "Methodology"]
)

with inputs_tab:
    st.subheader("Editable financial inputs")
    st.caption("Fetched values are starting points. Replace any figure that does not match the latest filing or your normalization judgment.")
    left, middle, right = st.columns(3)
    with left:
        price = num_input("Current share price", price_default, "Current market price per share.", "price", 0.1)
        diluted_shares = num_input("Diluted shares outstanding (mm)", shares_default, "Current or normalized diluted share count in millions.", "shares", 1.0)
        net_income = num_input("Normalized net income ($mm)", net_income_default, "Recurring net income attributable to common shareholders.", "net_income", 10.0)
        depreciation_amortization = num_input("Depreciation & amortization ($mm)", da_default, "Non-cash depreciation and amortization added back to net income.", "da", 10.0)
    with middle:
        other_non_cash = num_input("Other recurring non-cash charges ($mm)", other_non_cash_default, "Only recurring non-cash charges that should be added back.", "other_non_cash", 10.0)
        stock_based_compensation = num_input("Stock-based compensation ($mm)", sbc_default, "Economic cost of employee equity compensation. Deducting it is conservative.", "sbc", 10.0)
        deduct_sbc = st.toggle("Deduct stock-based compensation", value=True, help="When enabled, SBC reduces owner earnings.")
        maintenance_method = st.radio(
            "Maintenance capex method",
            ["Percentage of total capex", "Manual amount"],
            horizontal=True,
            help="Maintenance capex is the spending required to preserve current earning capacity, not total growth investment.",
        )
        if maintenance_method == "Percentage of total capex":
            maintenance_pct = st.slider("Maintenance capex percentage", 0.0, 1.5, 0.70, 0.05, help="A judgmental estimate applied to total capital expenditure.")
            maintenance_capex = capex_default * maintenance_pct
            st.caption(f"Estimated maintenance capex: {money(maintenance_capex)} million")
        else:
            maintenance_capex = num_input("Maintenance capex ($mm)", capex_default * 0.70, "Your direct estimate of maintenance capital expenditure.", "maintenance_capex", 10.0)
    with right:
        working_capital = num_input("Recurring working-capital investment ($mm)", working_capital_default, "Positive cash required to support operations. Enter zero when changes are temporary or non-recurring.", "working_capital", 10.0)
        excess_cash = num_input("Excess cash adjustment ($mm)", max(cash_default - 0.02 * value_or_zero(ltm.get("revenue")), 0.0), "Cash not required for normal operations. Added to equity value.", "excess_cash", 10.0)
        other_adjustments = num_input("Other equity adjustments ($mm)", 0.0, "Optional net adjustment. A common conservative approach is to subtract debt here only when owner earnings are modeled before financing; because this model starts from net income, default is zero in most cases.", "other_adjustments", 10.0)
        normalized_tax_rate = st.number_input("Normalized tax rate", 0.0, 0.60, 0.21, 0.01, format="%.1f", help="Used for NOPAT and ROIC calculations, not directly in owner earnings because net income is already after tax.")

    st.subheader("Valuation assumptions")
    a, b, c, d = st.columns(4)
    with a:
        start_growth = st.number_input("Starting owner earnings growth", -0.50, 0.50, 0.08, 0.005, format="%.3f", help="Growth applied in forecast year 1.")
        year_10_growth = st.number_input("Year 10 growth", -0.20, 0.20, 0.04, 0.005, format="%.3f", help="Growth fades linearly from year 1 to this rate in year 10.")
    with b:
        terminal_growth = st.number_input("Terminal growth", -0.05, 0.08, 0.025, 0.0025, format="%.4f", help="Perpetual growth after year 10. It must remain below the discount rate.")
        discount_rate = st.number_input("Required return / discount rate", 0.01, 0.30, 0.09, 0.005, format="%.3f", help="Your opportunity cost adjusted for business risk and forecast uncertainty.")
    with c:
        margin_target = st.number_input("Margin of safety target", 0.0, 0.80, 0.25, 0.05, format="%.2f", help="Target discount from intrinsic value to purchase price.")
        minimum_oe_yield = st.number_input("Minimum owner earnings yield", 0.01, 0.30, 0.06, 0.005, format="%.3f", help="Alternative price discipline: owner earnings per share divided by price.")
    with d:
        st.info("Use conservative assumptions. The terminal value often represents a large share of total value.")

# State-backed inputs are created in Inputs tab even when another tab is open because Streamlit executes all tabs.
inputs = ValuationInputs(
    price=price,
    diluted_shares_mm=diluted_shares,
    net_income_mm=net_income,
    depreciation_amortization_mm=depreciation_amortization,
    other_non_cash_mm=other_non_cash,
    stock_based_compensation_mm=stock_based_compensation,
    maintenance_capex_mm=maintenance_capex,
    working_capital_investment_mm=working_capital,
    excess_cash_mm=excess_cash,
    other_equity_adjustments_mm=other_adjustments,
    deduct_sbc=deduct_sbc,
    start_growth=start_growth,
    year_10_growth=year_10_growth,
    terminal_growth=terminal_growth,
    discount_rate=discount_rate,
    margin_of_safety_target=margin_target,
    minimum_owner_earnings_yield=minimum_oe_yield,
)

try:
    result = run_dcf(inputs)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

scenarios = {
    "Bear": {"start_growth": 0.03, "discount_rate": 0.11, "terminal_growth": 0.015, "probability": 0.25},
    "Base": {"start_growth": start_growth, "discount_rate": discount_rate, "terminal_growth": terminal_growth, "probability": 0.50},
    "Bull": {"start_growth": 0.11, "discount_rate": 0.08, "terminal_growth": 0.03, "probability": 0.25},
}
scenario_df = scenario_values(inputs, scenarios)
weighted_value = scenario_df["Weighted value"].sum()
quality_df = calculate_quality_metrics(historical, tax_rate=normalized_tax_rate) if not historical.empty else pd.DataFrame()

with summary_tab:
    cols = st.columns(5)
    cols[0].metric("Current price", money(price, company.currency))
    cols[1].metric("Owner earnings / share", money(result["owner_earnings_per_share"], company.currency))
    cols[2].metric("No-growth value", money(result["no_growth_value_per_share"], company.currency))
    cols[3].metric("DCF intrinsic value", money(result["intrinsic_value_per_share"], company.currency))
    cols[4].metric("Margin of safety", pct(result["margin_of_safety"]))

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.line(
            result["forecast"], x="Year", y="Owner earnings ($mm)", markers=True,
            title="Base-case owner earnings forecast",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), yaxis_title="$ millions")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Valuation checks")
        checks = pd.DataFrame(
            {
                "Measure": ["Base DCF", "Probability-weighted", "Target buy price", "Yield-based buy price"],
                "Value per share": [result["intrinsic_value_per_share"], weighted_value, result["target_buy_price"], result["yield_based_buy_price"]],
            }
        )
        st.dataframe(checks.style.format({"Value per share": "${:,.2f}"}), hide_index=True, use_container_width=True)
        st.caption(f"Current owner earnings yield: {pct(result['owner_earnings_yield'])} · Price / owner earnings: {result['price_to_owner_earnings']:.1f}x")

    st.subheader("Downloads")
    export_bytes = build_excel_export(
        {"name": company.name, "symbol": company.symbol, "price": company.price},
        historical,
        result,
        scenario_df,
    )
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Download populated analysis (.xlsx)", export_bytes, f"{company.symbol}_owner_earnings_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    dl2.download_button("Download original model template", TEMPLATE_PATH.read_bytes(), TEMPLATE_PATH.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    json_payload = {k: v for k, v in result.items() if k != "forecast"}
    json_payload["forecast"] = result["forecast"].to_dict(orient="records")
    dl3.download_button("Download model data (.json)", json.dumps(json_payload, indent=2, default=str), f"{company.symbol}_valuation.json", "application/json", use_container_width=True)

with quality_tab:
    st.subheader("Business quality")
    if quality_df.empty:
        st.info("Historical statements are not available for this ticker.")
    else:
        latest = quality_df.iloc[-1]
        metrics = st.columns(5)
        metrics[0].metric("Latest ROIC", pct(latest.get("roic")))
        metrics[1].metric("EBIT margin", pct(latest.get("ebit_margin")))
        metrics[2].metric("Net margin", pct(latest.get("net_margin")))
        metrics[3].metric("Interest coverage", "—" if pd.isna(latest.get("interest_coverage")) else f"{latest.get('interest_coverage'):.1f}x")
        metrics[4].metric("Share-count growth", pct(latest.get("share_count_growth")))

        display_cols = ["date", "revenue_growth", "ebit_margin", "net_margin", "roic", "roa", "roe", "interest_coverage", "share_count_growth"]
        shown = quality_df[[c for c in display_cols if c in quality_df.columns]].copy()
        st.dataframe(
            shown.style.format({
                "revenue_growth": "{:.1%}", "ebit_margin": "{:.1%}", "net_margin": "{:.1%}",
                "roic": "{:.1%}", "roa": "{:.1%}", "roe": "{:.1%}",
                "interest_coverage": "{:.1f}x", "share_count_growth": "{:.1%}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
        chart_data = quality_df[["date", "roic", "ebit_margin", "net_margin"]].melt("date", var_name="Metric", value_name="Value")
        fig = px.line(chart_data, x="date", y="Value", color="Metric", markers=True, title="Return and margin trends")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with valuation_tab:
    st.subheader("DCF forecast")
    st.dataframe(
        result["forecast"].style.format({"Growth rate": "{:.1%}", "Owner earnings ($mm)": "${:,.1f}", "Present value ($mm)": "${:,.1f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Scenario analysis")
    st.dataframe(
        scenario_df.style.format({"Intrinsic value per share": "${:,.2f}", "Probability": "{:.0%}", "Weighted value": "${:,.2f}", "Margin of safety": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.metric("Probability-weighted intrinsic value", money(weighted_value, company.currency))

with history_tab:
    st.subheader("Fetched historical statements")
    st.caption("Values are standardized by the selected data provider and should be reconciled with annual reports before making an investment decision.")
    if historical.empty:
        st.info("No historical data were returned.")
    else:
        display = historical.copy()
        st.dataframe(display.style.format(precision=1, na_rep="—"), hide_index=True, use_container_width=True)
        if {"date", "revenue", "net_income", "operating_cash_flow"}.issubset(display.columns):
            chart = display[["date", "revenue", "net_income", "operating_cash_flow"]].melt("date", var_name="Metric", value_name="$mm")
            fig = px.bar(chart, x="date", y="$mm", color="Metric", barmode="group", title="Historical financial performance")
            st.plotly_chart(fig, use_container_width=True)

with method_tab:
    st.subheader("Methodology")
    st.markdown(
        """
        **Owner earnings** estimate the cash that could be distributed to owners without impairing the company's current competitive position:

        `Net income + depreciation & amortization + recurring non-cash charges − stock-based compensation − maintenance capex − recurring working-capital investment`

        **No-growth earning power value** capitalizes normalized owner earnings as a perpetuity:

        `Owner earnings per share ÷ required return`

        **DCF intrinsic value** discounts ten years of forecast owner earnings and a terminal value back to the present. Growth fades linearly from the starting rate to the year-10 rate.

        **Required return** is an opportunity-cost assumption, not an observable accounting number. It should reflect available alternatives, business quality, leverage, cyclicality, and forecast uncertainty.

        **Maintenance capex** is usually the most judgmental input. Total capex may include growth investment and can therefore understate owner earnings when used mechanically.
        """
    )
    st.warning("This application automates arithmetic and data collection, not investment judgment. Always inspect filings, footnotes, acquisition accounting, dilution, debt terms, and capital-allocation history.")
