import pytest

from valuation import ValuationInputs, owner_earnings, run_dcf


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


def test_discount_rate_must_exceed_terminal_growth():
    inputs = sample_inputs()
    invalid = ValuationInputs(**{**inputs.__dict__, "discount_rate": 0.02, "terminal_growth": 0.02})
    with pytest.raises(ValueError):
        run_dcf(invalid)
