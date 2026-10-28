# NSW Statewide Hospital Utilisation Dashboard

Interactive portfolio prototype built with **Python, Dash and Plotly** for a University of Sydney QBUS5010 open-ended dashboard project.

> **Data note:** This public portfolio version uses mock/synthetic hospital data. The 85% capacity benchmark and calibrated occupancy logic are illustrative project assumptions and should not be interpreted as official NSW Health statistics.

## Project framing

The course used an open-ended brief. Our team independently selected a real-world healthcare resource-monitoring problem, translated it into dashboard requirements, and designed a working interactive prototype.

## What the dashboard includes

- Global filters for date, Local Health District, hospital tier, admission type and ICD-10 code
- KPI cards for infrastructure utilisation, patient encounters and average length of stay
- LHD bed-occupancy comparison against an illustrative 85% benchmark
- Monthly / quarterly hospital-activity trends
- Regional bottleneck map
- IPD average length-of-stay trend
- Patient age-cohort comparison
- Cross-filtered Dash callbacks so the views update together

## AI-assisted development

Generative AI was used as a development collaborator for requirement decomposition, code scaffolding, debugging and interaction refinement. Outputs were manually reviewed, tested and iterated rather than adopted without verification.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8050`.

## Deploy on Render

Upload this folder to a GitHub repository. Then either:

### Option A — Render Blueprint
Create a new Render Blueprint from the repository. Render will read `render.yaml`.

### Option B — Web Service
Use:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:server`

A Mapbox token is optional. If none is supplied, the app falls back to OpenStreetMap.

## Files

- `app.py` — Dash application
- `hospital_mockup_data.csv` — mock/synthetic project dataset
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deployment configuration
