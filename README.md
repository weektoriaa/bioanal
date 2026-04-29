# ASC Grouper (FLIM)

Small Streamlit web app for grouping `.asc` FLIM files and exporting Excel tables.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy as a website (GitHub + Streamlit Community Cloud)

1. Create a GitHub repository.
2. Upload/push this project (`app.py`, `requirements.txt`, `.streamlit/config.toml`).
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Click **New app** and select your GitHub repo.
5. Set main file path to `app.py` and deploy.

Your app will be available via a public web link.
