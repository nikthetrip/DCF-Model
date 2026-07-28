import pytest

from valuation import ValuationInputs, owner_earnings, run_dcf, sensitivity_table


def sample_inputs() -> ValuationInputs:
    return ValuationInputs(
        price=50,
        diluted_shares_mm=100,
        net_income_mm=600,
        depreciation_amortization_mm=100,
        other_non_cash_mm=20,
        stock_based_compensation_mm=30,
        maintenance_capex_mm=80,
        working_capital_investment_mm=10,
        discount_rate=0.10,
        terminal_growth=0.02,
    )


def test_owner_earnings():
    assert owner_earnings(sample_inputs()) == pytest.approx(600)


def test_dcf_returns_positive_value():
    result = run_dcf(sample_inputs())
    assert result["intrinsic_value_per_share"] > 0
    assert len(result["forecast"]) == 10
    assert 0 < result["terminal_value_share"] < 1


def test_discount_rate_must_exceed_terminal_growth():
    inputs = sample_inputs()
    invalid = ValuationInputs(**{**inputs.__dict__, "discount_rate": 0.02, "terminal_growth": 0.02})
    with pytest.raises(ValueError):
        run_dcf(invalid)


def test_non_operating_adjustments_flow_to_equity_value():
    base = run_dcf(sample_inputs())["equity_value_mm"]
    adjusted_inputs = ValuationInputs(**{
        **sample_inputs().__dict__,
        "excess_cash_mm": 100,
        "non_operating_assets_mm": 50,
        "other_claims_mm": 25,
    })
    adjusted = run_dcf(adjusted_inputs)["equity_value_mm"]
    assert adjusted - base == pytest.approx(125)


def test_sensitivity_table_shape_and_invalid_cells():
    table = sensitivity_table(sample_inputs(), [0.08, 0.10], [0.02, 0.09])
    assert table.shape == (2, 2)
    assert table.loc[0.02, "8.0%"] > 0
    assert table.loc[0.09, "8.0%"] != table.loc[0.09, "8.0%"]  # NaN
