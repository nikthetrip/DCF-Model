# Buffett–Munger Owner Earnings Streamlit App

A GitHub-ready Streamlit application that turns the Excel owner-earnings model into a simple web interface. Select a ticker, fetch market and financial-statement data, review editable inputs, and calculate owner earnings, no-growth earning power, a 10-year DCF, scenario values, quality metrics, and margin of safety.

## Features

- Searchable ticker workflow with a curated dropdown and custom-symbol override
- Yahoo Finance data without an API key
- Optional Financial Modeling Prep integration for standardized statements
- All automatically retrieved values remain editable
- English labels, explanatory tooltips, and a clean tabbed interface
- Owner earnings, no-growth value, DCF, bear/base/bull scenarios, and quality metrics
- Downloadable populated Excel analysis, JSON model data, and the original Excel template
- Caching, input validation, tests, and Streamlit deployment configuration

## Local installation

```bash
git clone <your-repository-url>
cd buffett-munger-owner-earnings
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data providers

### Yahoo Finance

Choose **Yahoo Finance (no key)** in the sidebar. This is convenient for prototypes and personal research, but some tickers can have missing or inconsistent statement rows.

### Financial Modeling Prep

Create an FMP API key and either paste it into the sidebar or configure it as a Streamlit secret:

```toml
# .streamlit/secrets.toml
FMP_API_KEY = "your_key_here"
```

For Streamlit Community Cloud, add the same value in the app's **Secrets** settings.

## Deploy on Streamlit Community Cloud

1. Push this directory to a public or private GitHub repository.
2. Open Streamlit Community Cloud and create a new app.
3. Select the repository, branch, and `app.py` entry point.
4. Add `FMP_API_KEY` in Secrets when using FMP.
5. Deploy.

## Important modeling judgments

The application automates data collection and arithmetic. It cannot determine the correct maintenance capex, normalized working-capital requirement, sustainable growth rate, discount rate, or competitive durability. Reconcile all data with annual reports and footnotes.

## Project structure

```text
.
├── app.py                 # Streamlit user interface
├── data_provider.py       # Yahoo Finance and FMP adapters
├── valuation.py           # Owner earnings, DCF, scenarios, quality metrics
├── export_model.py        # Downloadable Excel analysis
├── tickers.csv            # Curated dropdown list
├── assets/
│   └── Buffett_Munger_Owner_Earnings_Model.xlsx
├── tests/
│   └── test_valuation.py
├── requirements.txt
└── .streamlit/config.toml
```

## Run tests

```bash
pytest -q
```

## Disclaimer

For educational and research purposes only. This is not investment advice.
