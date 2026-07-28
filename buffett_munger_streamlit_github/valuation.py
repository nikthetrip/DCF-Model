from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValuationInputs:
    price: float
    diluted_shares_mm: float
    net_income_mm: float
    depreciation_amortization_mm: float
    other_non_cash_mm: float
    stock_based_compensation_mm: float
    maintenance_capex_mm: float
    working_capital_investment_mm: float
    excess_cash_mm: float = 0.0
    other_equity_adjustments_mm: float = 0.0
    deduct_sbc: bool = True
    start_growth: float = 0.08
    year_10_growth: float = 0.04
    terminal_growth: float = 0.025
    discount_rate: float = 0.09
    margin_of_safety_target: float = 0.25
    minimum_owner_earnings_yield: float = 0.06

    def validate(self) -> None:
        if self.diluted_shares_mm <= 0:
            raise ValueError("Diluted shares outstanding must be greater than zero.")
        if self.price < 0:
            raise ValueError("Share price cannot be negative.")
        if self.discount_rate <= self.terminal_growth:
            raise ValueError("Discount rate must be greater than terminal growth.")
        if not 0 <= self.margin_of_safety_target < 1:
            raise ValueError("Margin of safety must be between 0% and 100%.")
        if self.minimum_owner_earnings_yield <= 0:
            raise ValueError("Minimum owner earnings yield must be greater than zero.")


def owner_earnings(inputs: ValuationInputs) -> float:
    sbc_deduction = inputs.stock_based_compensation_mm if inputs.deduct_sbc else 0.0
    return (
        inputs.net_income_mm
        + inputs.depreciation_amortization_mm
        + inputs.other_non_cash_mm
        - sbc_deduction
        - inputs.maintenance_capex_mm
        - inputs.working_capital_investment_mm
    )


def forecast_growth_path(start_growth: float, year_10_growth: float, years: int = 10) -> np.ndarray:
    return np.linspace(start_growth, year_10_growth, years)


def run_dcf(inputs: ValuationInputs, years: int = 10) -> dict[str, Any]:
    inputs.validate()
    starting_oe = owner_earnings(inputs)
    growth_rates = forecast_growth_path(inputs.start_growth, inputs.year_10_growth, years)

    forecasts: list[float] = []
    present_values: list[float] = []
    current = starting_oe
    for year, growth in enumerate(growth_rates, start=1):
        current *= 1 + growth
        forecasts.append(current)
        present_values.append(current / ((1 + inputs.discount_rate) ** year))

    terminal_value = forecasts[-1] * (1 + inputs.terminal_growth) / (
        inputs.discount_rate - inputs.terminal_growth
    )
    terminal_pv = terminal_value / ((1 + inputs.discount_rate) ** years)
    equity_value_mm = (
        sum(present_values)
        + terminal_pv
        + inputs.excess_cash_mm
        + inputs.other_equity_adjustments_mm
    )
    intrinsic_value_per_share = equity_value_mm / inputs.diluted_shares_mm
    oe_per_share = starting_oe / inputs.diluted_shares_mm
    no_growth_value = oe_per_share / inputs.discount_rate
    target_buy_price = intrinsic_value_per_share * (1 - inputs.margin_of_safety_target)
    owner_earnings_yield = oe_per_share / inputs.price if inputs.price > 0 else np.nan
    price_to_owner_earnings = inputs.price / oe_per_share if oe_per_share > 0 else np.nan
    yield_based_buy_price = oe_per_share / inputs.minimum_owner_earnings_yield
    margin_of_safety = (
        (intrinsic_value_per_share - inputs.price) / intrinsic_value_per_share
        if intrinsic_value_per_share > 0
        else np.nan
    )

    forecast_table = pd.DataFrame(
        {
            "Year": range(1, years + 1),
            "Growth rate": growth_rates,
            "Owner earnings ($mm)": forecasts,
            "Present value ($mm)": present_values,
        }
    )

    return {
        "inputs": asdict(inputs),
        "owner_earnings_mm": starting_oe,
        "owner_earnings_per_share": oe_per_share,
        "owner_earnings_yield": owner_earnings_yield,
        "price_to_owner_earnings": price_to_owner_earnings,
        "no_growth_value_per_share": no_growth_value,
        "terminal_value_mm": terminal_value,
        "terminal_present_value_mm": terminal_pv,
        "equity_value_mm": equity_value_mm,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "target_buy_price": target_buy_price,
        "yield_based_buy_price": yield_based_buy_price,
        "margin_of_safety": margin_of_safety,
        "forecast": forecast_table,
    }


def scenario_values(
    base_inputs: ValuationInputs,
    scenarios: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, values in scenarios.items():
        scenario_input = ValuationInputs(
            **{
                **asdict(base_inputs),
                "start_growth": values["start_growth"],
                "discount_rate": values["discount_rate"],
                "terminal_growth": values["terminal_growth"],
            }
        )
        result = run_dcf(scenario_input)
        probability = values["probability"]
        rows.append(
            {
                "Scenario": name,
                "Intrinsic value per share": result["intrinsic_value_per_share"],
                "Probability": probability,
                "Weighted value": result["intrinsic_value_per_share"] * probability,
                "Margin of safety": result["margin_of_safety"],
            }
        )
    return pd.DataFrame(rows)


def safe_cagr(beginning: float, ending: float, periods: int) -> float:
    if periods <= 0 or beginning <= 0 or ending <= 0:
        return np.nan
    return (ending / beginning) ** (1 / periods) - 1


def calculate_quality_metrics(historical: pd.DataFrame, tax_rate: float = 0.21) -> pd.DataFrame:
    if historical.empty:
        return historical.copy()
    df = historical.copy().sort_values("date")
    df["revenue_growth"] = df["revenue"].pct_change()
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["net_margin"] = df["net_income"] / df["revenue"]
    df["nopat"] = df["ebit"] * (1 - tax_rate)
    avg_invested_capital = (df["invested_capital"] + df["invested_capital"].shift(1)) / 2
    df["roic"] = df["nopat"] / avg_invested_capital
    df["roa"] = df["net_income"] / ((df["total_assets"] + df["total_assets"].shift(1)) / 2)
    df["roe"] = df["net_income"] / ((df["equity"] + df["equity"].shift(1)) / 2)
    df["interest_coverage"] = df["ebit"] / df["interest_expense"].replace(0, np.nan)
    df["share_count_growth"] = df["diluted_shares"].pct_change()
    return df
