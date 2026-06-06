# Mobile Conversion Intent API (Dual-Engine)

A context-aware machine learning web service built to solve the e-commerce "Cold Start" problem. This API intercepts live frontend traffic, analyzes the behavioral context of the user, and dynamically routes the data to the optimal machine learning engine to predict purchase intent.

## The Architecture

This system operates a dynamic routing matrix (The Traffic Cop) using two frozen `.joblib` pipelines:

* **Engine A: The Greeter (Top of Funnel):** Triggers at "Millisecond Zero". When a user has zero page views, this engine relies exclusively on Day-One categorical context (Browser, Traffic Source, OS) using Target Encoding to predict base conversion probability.
* **Engine B: The Closer (Bottom of Funnel):** Takes over the moment a user begins generating behavioral data (clicks, page durations). It leverages a heavy-duty LightGBM pipeline to evaluate deep session engagement.

## Repository Structure

```text
├── data/raw/             # Immutable truth (Ignored in Docker build)
├── notebooks/            # Exploratory Lab (Ignored in Docker build)
├── src/
│   └── app.py            # FastAPI Routing & Production Logic
├── models/
│   ├── greeter_engine_v1.joblib
│   └── conversion_engine_v1.joblib
├── Dockerfile            # Debian Python:3.11-slim with libgomp1 patch
├── .dockerignore         # Build shield for optimal container weight
└── requirements.txt      # Deterministic production dependencies
```

## Local Development (Lab Environment)

To boot the API locally via Uvicorn for QA testing:

1. Activate your virtual environment.
2. Run the server:
   ```bash
   uvicorn src.app:app --reload
   ```
3. Access the interactive Swagger UI at: `http://127.0.0.1:8000/docs`

## Docker Deployment (Live Environment)

This API is packaged into a highly optimized, production-ready container. 

1. **Build the Image:**
   ```bash
   docker build -t conversion-api:v2 .
   ```
2. **Run the Container:**
   ```bash
   docker run -p 8000:8000 conversion-api:v2
   ```

##  API Schema & Routing Example

The `/predict` endpoint expects a unified JSON payload. Behavioral metrics default to `0` or `0.0` to minimize frontend payload size at initial landing.

**Sample Request (Millisecond Zero):**
```json
{
  "VisitorType": "New_Visitor",
  "TrafficType": 2,
  "Browser": 1,
  "OperatingSystems": 2,
  "Region": 1,
  "Month": "Feb",
  "Weekend": false
}
```

**Diagnostic Response:**
```json
{
  "engine_used": "Greeter Engine (Cold Start)",
  "conversion_probability": 0.1245,
  "high_intent_flag": false
}
```
