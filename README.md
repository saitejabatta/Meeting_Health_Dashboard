# Meeting Health Dashboard

Local runnable files for VS Code.

## Files

- `notebooks/Meeting_Health_Dashboard_fixed.ipynb`: runs the meeting intelligence pipeline.
- `dashboard/app.py`: runs the Streamlit dashboard.
- `dashboard/data/`: generated CSVs go here after the notebook runs.
- `requirements.txt`: Python dependencies.

## Setup

Open this folder in VS Code:

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local"
python3 -m pip install -r requirements.txt
```

Set your API key in the terminal before running the notebook or dashboard:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can also put the key in a local `.env` file at the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

## Run The Notebook

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local/notebooks"
jupyter notebook
```

Open `Meeting_Health_Dashboard_fixed.ipynb` and run all cells.

The notebook also supports a faster demo run by default. For full-length audio processing:

```bash
export FULL_MEETING_RUN=1
```

## Run The Dashboard

In a second terminal:

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local/dashboard"
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

Open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Notes

- The dashboard loads generated files automatically from `dashboard/data`.
- Manual uploads are optional overrides only.
- Keep your API key in the terminal environment. Do not paste it into code.
