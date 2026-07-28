# Owner Earnings Valuation — Streamlit

A Buffett–Munger inspired owner-earnings valuation app built with Streamlit.

## What changed in v2

- Theme-safe Overview metrics: no white-on-white cards in dark mode.
- Clear separation between **fetched financial facts**, **normalization judgments**, and **valuation assumptions**.
- Provider `other non-cash items` are reference-only and default to **zero add-back**.
- Maintenance capex explicitly shows total capex and the selected maintenance assumption.
- Working-capital normalization supports manual, LTM, 3-year and 5-year heuristic views.
- Reported LTM net income and normalized net income are separate fields.
- Explicit excess cash, non-operating assets and other senior claims adjustments.
- Data-completeness panel and provider warnings.
- Editable historical table feeding quality metrics.
- Expanded quality diagnostics: ROIC, FCF margin, OCF conversion, debt/FCF, interest coverage, dilution and goodwill intensity.
- Editable bear/base/bull scenarios.
- DCF sensitivity matrix for discount rate vs terminal growth.
- Corrected Yahoo Finance EPS scaling (EPS is per share and must not be divided by one million).
- FCF fallback to operating cash flow less total capex when a direct FCF field is missing.

## Repository structure

```text
app.py
data_provider.py
valuation.py
export_model.py
requirements.txt
tickers.csv
assets/
  Buffett_Munger_Owner_Earnings_Model.xlsx
tests/
  test_valuation.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

If this folder is inside another repository folder, set the **Main file path** to the full path, for example:

```text
buffett_munger_streamlit_github/app.py
```

If `app.py` is at the repository root, use simply:

```text
app.py
```

## Financial Modeling Prep secret

Do not commit API keys. In Streamlit Community Cloud, add:

```toml
FMP_API_KEY = "your_key_here"
```

under the app's Secrets settings.

## Important methodology note

The app automates data collection and valuation arithmetic. It does not determine maintenance capex, normalized working-capital needs or sustainable growth for you. Those require review of filings, footnotes and business economics.
