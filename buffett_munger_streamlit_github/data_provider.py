from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf


@dataclass
class CompanyData:
    symbol: str
    name: str
    currency: str
    price: float | None
    market_cap_mm: float | None
    historical: pd.DataFrame
    ltm: dict[str, float | None]
    source: str
    warnings: list[str]


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "ebit": ("Operating Income", "EBIT"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "pretax_income": ("Pretax Income", "Income Before Tax"),
    "tax_expense": ("Tax Provision", "Income Tax Expense"),
    "net_income": ("Net Income Common Stockholders", "Net Income"),
    "diluted_eps": ("Diluted EPS",),
    "diluted_shares": ("Diluted Average Shares", "Diluted Shares"),
    "depreciation_amortization": ("Depreciation And Amortization", "Depreciation Amortization Depletion"),
    "stock_based_compensation": ("Stock Based Compensation",),
    "other_non_cash": ("Other Non Cash Items",),
    "capital_expenditure": ("Capital Expenditure", "Capital Expenditures"),
    "change_in_working_capital": ("Change In Working Capital",),
    "operating_cash_flow": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "free_cash_flow": ("Free Cash Flow",),
    "cash": ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"),
    "total_debt": ("Total Debt",),
    "total_assets": ("Total Assets",),
    "equity": ("Stockholders Equity", "Total Stockholder Equity"),
    "invested_capital": ("Invested Capital",),
    "goodwill_intangibles": ("Goodwill And Other Intangible Assets", "Goodwill"),
}

# Most Yahoo statement values are absolute currency/share-count units and need to be
# converted to millions. EPS is already a per-share number and must NOT be divided.
FIELD_DIVISORS: dict[str, float] = {field: 1_000_000 for field in FIELD_ALIASES}
FIELD_DIVISORS["diluted_eps"] = 1.0


def _first_existing(frame: pd.DataFrame, aliases: Iterable[str], column: Any, divisor: float) -> float | None:
    for alias in aliases:
        if alias in frame.index and column in frame.columns:
            value = frame.at[alias, column]
            if pd.notna(value):
                return float(value) / divisor
    return None


def _statement_value(frame: pd.DataFrame, field: str, column: Any) -> float | None:
    if frame is None or frame.empty:
        return None
    return _first_existing(frame, FIELD_ALIASES[field], column, FIELD_DIVISORS[field])


def _coalesce_fcf(ocf: float | None, capex: float | None, direct_fcf: float | None) -> float | None:
    if direct_fcf is not None:
        return direct_fcf
    if ocf is not None and capex is not None:
        return ocf - capex
    return None


def _build_yfinance_rows(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(income.columns) | set(balance.columns) | set(cashflow.columns))
    rows: list[dict[str, Any]] = []
    for column in columns:
        capex = _statement_value(cashflow, "capital_expenditure", column)
        if capex is not None:
            capex = abs(capex)
        ocf = _statement_value(cashflow, "operating_cash_flow", column)
        direct_fcf = _statement_value(cashflow, "free_cash_flow", column)
        interest = _statement_value(income, "interest_expense", column)
        row = {
            "date": pd.Timestamp(column).date(),
            "revenue": _statement_value(income, "revenue", column),
            "ebit": _statement_value(income, "ebit", column),
            "interest_expense": abs(interest) if interest is not None else None,
            "pretax_income": _statement_value(income, "pretax_income", column),
            "tax_expense": _statement_value(income, "tax_expense", column),
            "net_income": _statement_value(income, "net_income", column),
            "diluted_eps": _statement_value(income, "diluted_eps", column),
            "diluted_shares": _statement_value(income, "diluted_shares", column),
            "depreciation_amortization": _statement_value(cashflow, "depreciation_amortization", column),
            "stock_based_compensation": _statement_value(cashflow, "stock_based_compensation", column),
            "other_non_cash": _statement_value(cashflow, "other_non_cash", column),
            "capital_expenditure": capex,
            "change_in_working_capital": _statement_value(cashflow, "change_in_working_capital", column),
            "operating_cash_flow": ocf,
            "free_cash_flow": _coalesce_fcf(ocf, capex, direct_fcf),
            "cash": _statement_value(balance, "cash", column),
            "total_debt": _statement_value(balance, "total_debt", column),
            "total_assets": _statement_value(balance, "total_assets", column),
            "equity": _statement_value(balance, "equity", column),
            "invested_capital": _statement_value(balance, "invested_capital", column),
            "goodwill_intangibles": _statement_value(balance, "goodwill_intangibles", column),
        }
        if row["invested_capital"] is None and all(row.get(k) is not None for k in ["total_debt", "equity", "cash"]):
            row["invested_capital"] = row["total_debt"] + row["equity"] - row["cash"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").tail(5).reset_index(drop=True)


def _missing_field_warnings(ltm: dict[str, float | None]) -> list[str]:
    critical = {
        "revenue": "Revenue",
        "net_income": "Net income",
        "diluted_shares": "Diluted shares",
        "depreciation_amortization": "Depreciation & amortization",
        "capital_expenditure": "Capital expenditure",
        "operating_cash_flow": "Operating cash flow",
        "total_debt": "Total debt",
    }
    missing = [label for key, label in critical.items() if ltm.get(key) is None]
    if not missing:
        return []
    return ["Missing provider fields: " + ", ".join(missing) + ". Enter or verify them manually."]


def fetch_yfinance(symbol: str) -> CompanyData:
    ticker = yf.Ticker(symbol)
    info = ticker.get_info()
    historical = _build_yfinance_rows(ticker.income_stmt, ticker.balance_sheet, ticker.cashflow)

    q_income = ticker.quarterly_income_stmt
    q_balance = ticker.quarterly_balance_sheet
    q_cash = ticker.quarterly_cashflow
    warnings: list[str] = []

    if q_income.empty or q_cash.empty:
        warnings.append("Quarterly statements were unavailable; LTM values use the latest annual period.")
        ltm = historical.iloc[-1].to_dict() if not historical.empty else {}
    else:
        q_columns = list(q_income.columns[:4])
        if len(q_columns) < 4:
            warnings.append(f"Only {len(q_columns)} quarterly income periods were available; LTM may be incomplete.")

        def qsum(frame: pd.DataFrame, field: str, columns: list[Any]) -> float | None:
            vals = [_statement_value(frame, field, col) for col in columns if col in frame.columns]
            nums = [v for v in vals if v is not None]
            return sum(nums) if nums else None

        latest_q = q_balance.columns[0] if not q_balance.empty else None
        cash_columns = list(q_cash.columns[:4])
        capex = qsum(q_cash, "capital_expenditure", cash_columns)
        capex = abs(capex) if capex is not None else None
        ocf = qsum(q_cash, "operating_cash_flow", cash_columns)
        direct_fcf = qsum(q_cash, "free_cash_flow", cash_columns)
        interest = qsum(q_income, "interest_expense", q_columns)
        ltm = {
            "revenue": qsum(q_income, "revenue", q_columns),
            "ebit": qsum(q_income, "ebit", q_columns),
            "interest_expense": abs(interest) if interest is not None else None,
            "pretax_income": qsum(q_income, "pretax_income", q_columns),
            "tax_expense": qsum(q_income, "tax_expense", q_columns),
            "net_income": qsum(q_income, "net_income", q_columns),
            "diluted_eps": qsum(q_income, "diluted_eps", q_columns),
            "diluted_shares": _statement_value(q_income, "diluted_shares", q_columns[0]) if q_columns else None,
            "depreciation_amortization": qsum(q_cash, "depreciation_amortization", cash_columns),
            "stock_based_compensation": qsum(q_cash, "stock_based_compensation", cash_columns),
            "other_non_cash": qsum(q_cash, "other_non_cash", cash_columns),
            "capital_expenditure": capex,
            "change_in_working_capital": qsum(q_cash, "change_in_working_capital", cash_columns),
            "operating_cash_flow": ocf,
            "free_cash_flow": _coalesce_fcf(ocf, capex, direct_fcf),
            "cash": _statement_value(q_balance, "cash", latest_q) if latest_q is not None else None,
            "total_debt": _statement_value(q_balance, "total_debt", latest_q) if latest_q is not None else None,
            "total_assets": _statement_value(q_balance, "total_assets", latest_q) if latest_q is not None else None,
            "equity": _statement_value(q_balance, "equity", latest_q) if latest_q is not None else None,
            "invested_capital": _statement_value(q_balance, "invested_capital", latest_q) if latest_q is not None else None,
            "goodwill_intangibles": _statement_value(q_balance, "goodwill_intangibles", latest_q) if latest_q is not None else None,
        }
        if ltm["invested_capital"] is None and all(ltm.get(k) is not None for k in ["total_debt", "equity", "cash"]):
            ltm["invested_capital"] = ltm["total_debt"] + ltm["equity"] - ltm["cash"]

    warnings.extend(_missing_field_warnings(ltm))
    if ltm.get("other_non_cash") not in (None, 0):
        warnings.append("Provider 'other non-cash items' are shown for reference only and are NOT added to owner earnings by default. Review the filing before using them.")

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    return CompanyData(
        symbol=symbol.upper(),
        name=info.get("longName") or info.get("shortName") or symbol.upper(),
        currency=info.get("currency") or "USD",
        price=float(price) if price is not None else None,
        market_cap_mm=float(market_cap) / 1_000_000 if market_cap else None,
        historical=historical,
        ltm=ltm,
        source="Yahoo Finance via yfinance",
        warnings=warnings,
    )


class FMPClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("An FMP API key is required.")
        self.base_url = "https://financialmodelingprep.com/stable"

    def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/{endpoint}", params={**params, "apikey": self.api_key}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise RuntimeError(payload["Error Message"])
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected FMP response for {endpoint}.")
        return payload


def _fmp_num(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return float(value) / 1_000_000
    return None


def _fmp_row(inc: dict[str, Any], bal: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    capex_raw = _fmp_num(cf, "capitalExpenditure", "investmentsInPropertyPlantAndEquipment")
    capex = abs(capex_raw) if capex_raw is not None else None
    cash = _fmp_num(bal, "cashAndCashEquivalents", "cashAndShortTermInvestments")
    debt = _fmp_num(bal, "totalDebt")
    equity = _fmp_num(bal, "totalStockholdersEquity", "totalEquity")
    invested = _fmp_num(bal, "investedCapital")
    if invested is None and all(v is not None for v in [debt, equity, cash]):
        invested = debt + equity - cash
    ocf = _fmp_num(cf, "operatingCashFlow", "netCashProvidedByOperatingActivities")
    direct_fcf = _fmp_num(cf, "freeCashFlow")
    interest = _fmp_num(inc, "interestExpense")
    return {
        "date": pd.to_datetime(inc.get("date")).date(),
        "revenue": _fmp_num(inc, "revenue"),
        "ebit": _fmp_num(inc, "operatingIncome", "ebit"),
        "interest_expense": abs(interest) if interest is not None else None,
        "pretax_income": _fmp_num(inc, "incomeBeforeTax"),
        "tax_expense": _fmp_num(inc, "incomeTaxExpense"),
        "net_income": _fmp_num(inc, "netIncome"),
        "diluted_eps": float(inc.get("epsDiluted")) if inc.get("epsDiluted") is not None else None,
        "diluted_shares": _fmp_num(inc, "weightedAverageShsOutDil"),
        "depreciation_amortization": _fmp_num(cf, "depreciationAndAmortization"),
        "stock_based_compensation": _fmp_num(cf, "stockBasedCompensation"),
        "other_non_cash": _fmp_num(cf, "otherNonCashItems"),
        "capital_expenditure": capex,
        "change_in_working_capital": _fmp_num(cf, "changeInWorkingCapital"),
        "operating_cash_flow": ocf,
        "free_cash_flow": _coalesce_fcf(ocf, capex, direct_fcf),
        "cash": cash,
        "total_debt": debt,
        "total_assets": _fmp_num(bal, "totalAssets"),
        "equity": equity,
        "invested_capital": invested,
        "goodwill_intangibles": _fmp_num(bal, "goodwillAndIntangibleAssets"),
    }


def fetch_fmp(symbol: str, api_key: str | None = None) -> CompanyData:
    client = FMPClient(api_key)
    profile = client.get("profile", symbol=symbol)
    quote = client.get("quote", symbol=symbol)
    annual_income = client.get("income-statement", symbol=symbol, period="annual", limit=5)
    annual_balance = client.get("balance-sheet-statement", symbol=symbol, period="annual", limit=5)
    annual_cash = client.get("cash-flow-statement", symbol=symbol, period="annual", limit=5)
    quarterly_income = client.get("income-statement", symbol=symbol, period="quarter", limit=4)
    quarterly_balance = client.get("balance-sheet-statement", symbol=symbol, period="quarter", limit=1)
    quarterly_cash = client.get("cash-flow-statement", symbol=symbol, period="quarter", limit=4)

    balance_by_date = {x.get("date"): x for x in annual_balance}
    cash_by_date = {x.get("date"): x for x in annual_cash}
    rows = [_fmp_row(inc, balance_by_date.get(inc.get("date"), {}), cash_by_date.get(inc.get("date"), {})) for inc in annual_income]
    historical = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
    warnings: list[str] = []

    if quarterly_income and quarterly_cash:
        q_balance = quarterly_balance[0] if quarterly_balance else {}

        def sum_key(records: list[dict[str, Any]], *keys: str) -> float | None:
            values = [_fmp_num(record, *keys) for record in records]
            nums = [v for v in values if v is not None]
            return sum(nums) if nums else None

        capex_raw = sum_key(quarterly_cash, "capitalExpenditure")
        capex = abs(capex_raw) if capex_raw is not None else None
        ocf = sum_key(quarterly_cash, "operatingCashFlow", "netCashProvidedByOperatingActivities")
        direct_fcf = sum_key(quarterly_cash, "freeCashFlow")
        interest = sum_key(quarterly_income, "interestExpense")
        ltm = {
            "revenue": sum_key(quarterly_income, "revenue"),
            "ebit": sum_key(quarterly_income, "operatingIncome", "ebit"),
            "interest_expense": abs(interest) if interest is not None else None,
            "pretax_income": sum_key(quarterly_income, "incomeBeforeTax"),
            "tax_expense": sum_key(quarterly_income, "incomeTaxExpense"),
            "net_income": sum_key(quarterly_income, "netIncome"),
            "diluted_eps": sum(float(x.get("epsDiluted") or 0) for x in quarterly_income),
            "diluted_shares": _fmp_num(quarterly_income[0], "weightedAverageShsOutDil"),
            "depreciation_amortization": sum_key(quarterly_cash, "depreciationAndAmortization"),
            "stock_based_compensation": sum_key(quarterly_cash, "stockBasedCompensation"),
            "other_non_cash": sum_key(quarterly_cash, "otherNonCashItems"),
            "capital_expenditure": capex,
            "change_in_working_capital": sum_key(quarterly_cash, "changeInWorkingCapital"),
            "operating_cash_flow": ocf,
            "free_cash_flow": _coalesce_fcf(ocf, capex, direct_fcf),
            "cash": _fmp_num(q_balance, "cashAndCashEquivalents", "cashAndShortTermInvestments"),
            "total_debt": _fmp_num(q_balance, "totalDebt"),
            "total_assets": _fmp_num(q_balance, "totalAssets"),
            "equity": _fmp_num(q_balance, "totalStockholdersEquity", "totalEquity"),
            "invested_capital": _fmp_num(q_balance, "investedCapital"),
            "goodwill_intangibles": _fmp_num(q_balance, "goodwillAndIntangibleAssets"),
        }
        if ltm["invested_capital"] is None and all(ltm.get(k) is not None for k in ["total_debt", "equity", "cash"]):
            ltm["invested_capital"] = ltm["total_debt"] + ltm["equity"] - ltm["cash"]
    else:
        warnings.append("Quarterly statements were unavailable; LTM values use the latest annual period.")
        ltm = historical.iloc[-1].to_dict() if not historical.empty else {}

    warnings.extend(_missing_field_warnings(ltm))
    if ltm.get("other_non_cash") not in (None, 0):
        warnings.append("Provider 'other non-cash items' are shown for reference only and are NOT added to owner earnings by default. Review the filing before using them.")

    p = profile[0] if profile else {}
    q = quote[0] if quote else {}
    price = q.get("price") or p.get("price")
    market_cap = q.get("marketCap") or p.get("marketCap")
    return CompanyData(
        symbol=symbol.upper(),
        name=p.get("companyName") or p.get("name") or symbol.upper(),
        currency=p.get("currency") or "USD",
        price=float(price) if price is not None else None,
        market_cap_mm=float(market_cap) / 1_000_000 if market_cap else None,
        historical=historical,
        ltm=ltm,
        source="Financial Modeling Prep",
        warnings=warnings,
    )
