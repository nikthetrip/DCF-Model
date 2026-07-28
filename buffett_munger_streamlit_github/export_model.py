from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def build_excel_export(
    company: dict[str, Any],
    historical: pd.DataFrame,
    valuation_result: dict[str, Any],
    scenarios: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78"})
        money = workbook.add_format({"num_format": "$#,##0.00;[Red]($#,##0.00);-"})
        money_mm = workbook.add_format({"num_format": "$#,##0.0;[Red]($#,##0.0);-"})
        percent = workbook.add_format({"num_format": "0.0%;[Red](0.0%);-"})

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Company", "Ticker", "Current price", "Owner earnings ($mm)",
                    "Owner earnings per share", "Owner earnings yield",
                    "No-growth value per share", "DCF intrinsic value per share",
                    "Target buy price", "Margin of safety",
                ],
                "Value": [
                    company.get("name"), company.get("symbol"), company.get("price"),
                    valuation_result["owner_earnings_mm"], valuation_result["owner_earnings_per_share"],
                    valuation_result["owner_earnings_yield"], valuation_result["no_growth_value_per_share"],
                    valuation_result["intrinsic_value_per_share"], valuation_result["target_buy_price"],
                    valuation_result["margin_of_safety"],
                ],
            }
        )
        summary.to_excel(writer, sheet_name="Dashboard", index=False)
        historical.to_excel(writer, sheet_name="Historical Data", index=False)
        valuation_result["forecast"].to_excel(writer, sheet_name="DCF", index=False)
        scenarios.to_excel(writer, sheet_name="Scenarios", index=False)
        pd.DataFrame([valuation_result["inputs"]]).to_excel(writer, sheet_name="Assumptions", index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.hide_gridlines(2)
            ws.freeze_panes(1, 0)
            ws.set_row(0, 22, header)
            ws.set_column(0, 0, 32)
            ws.set_column(1, 30, 18)
        writer.sheets["Dashboard"].set_column("B:B", 20)
        writer.sheets["Dashboard"].set_column("B:B", 20, money)
        writer.sheets["Historical Data"].set_column("B:Z", 16, money_mm)
        writer.sheets["DCF"].set_column("B:B", 14, percent)
        writer.sheets["DCF"].set_column("C:D", 20, money_mm)
        writer.sheets["Scenarios"].set_column("B:B", 24, money)
        writer.sheets["Scenarios"].set_column("C:C", 14, percent)
        writer.sheets["Scenarios"].set_column("D:D", 20, money)
        writer.sheets["Scenarios"].set_column("E:E", 16, percent)
    return output.getvalue()
