# Iron Condor Streamlit Desk

A minimal Streamlit app for exploring an options iron condor using Black–Scholes pricing and greeks.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Quick start (Streamlit Cloud)

1. Push this repository to GitHub.
2. In Streamlit Community Cloud choose **New app** and set the fields exactly as follows:
   - Repository: `<user>/<repo>`
   - Branch: `main`
   - Main file path: `app.py`
3. Deploy and enjoy the interactive desk.
