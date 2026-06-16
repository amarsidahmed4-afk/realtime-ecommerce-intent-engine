# ⚡ Realtime Ecommerce Intent Engine

A high-performance, context-aware machine learning microservice built to optimize real-time e-commerce conversion pipelines. This service intercepts live web traffic telemetry via Google Tag Manager (GTM), analyzes the behavioral context of the customer journey, and dynamically routes payloads to specialized machine learning engines to predict purchase intent in milliseconds.

## 🏗️ The 4-Pillar Architecture

The infrastructure is designed as a decoupled, serverless ecosystem built for infinite scale and zero idle-cost compute:

1. **Ingestion (Google Tag Manager):** Captures high-intent customer behavior on the frontend via a low-latency Data Layer without touching the core website codebase.
2. **Infrastructure (Docker & Google Cloud Run):** Containerized using Docker to eliminate environment conflict, deployed as a stateless microservice that auto-scales to handle massive concurrent traffic spikes.
3. **The ML Brain (FastAPI & LightGBM):** Runs an asynchronous FastAPI web server that passes incoming web traffic through Pydantic validators and custom Target Encoders, executing rapid-fire inferences via specialized models.
4. **Observability (Streamlit & BigQuery):** Logs every single inference call in real-time to a permanent Google BigQuery data warehouse for auditability and drift detection, while serving an interactive Streamlit frontend for non-technical stakeholders.

---

## 🔀 Dynamic Routing Matrix

To maximize performance across different stages of the online customer journey, the API executes a custom routing matrix across two pre-trained `.joblib` pipelines:

* **The Greeter Engine (Top of Funnel):** Activates at "Millisecond Zero" when a user first lands on the site (0 product page views). It relies exclusively on day-one categorical data (Browser, Traffic Source, OS, Region) using pre-fitted Target Encoders to predict baseline conversion propensity.
* **The Closer Engine (Bottom of Funnel):** Automatically takes over the moment a user views a single product page. It evaluates deep session engagement metrics (Page Values, Exit Rates, durations) using a heavily optimized LightGBM classifier.

> 📊 **Business Logic Optimization:** Rather than relying on standard, unoptimized default decision boundaries (50%), this engine applies a strict, hardcoded threshold of **70%**—discovered via 200 Optuna optimization trials—maximizing the F1-Score (0.678) to shield businesses from wasting ad spend on false positives.

---

## 🗂️ Repository Structure

```text
├── models/               # Serialized mathematical brains (.joblib)
│   ├── greeter_engine_v1.joblib
│   └── conversion_engine_v1.joblib
├── notebooks/            # Laboratory: Optuna Optimization & Data Exploration
├── src/                  # The Production Factory Floor
│   ├── app.py            # FastAPI Application & Ingestion Endpoints
│   ├── marketing_pipeline.py    # Target Encoding & Data Translation Layer
│   ├── marketing_evaluation.py  # F1-Score & Threshold Thresholding Math
│   └── marketing_eda.py         # Automated exploratory data helpers
├── Dockerfile            # Containerization recipe for Google Cloud Run
├── streamlit_app.py      # Visual Showroom: Interactive client simulator
├── requirements.txt      # Production backend dependencies
├── requirements-ui.txt   # Frontend UI dependencies
└── requirements-dev.txt  # Local data science lab packages
```

---

## 🚀 Execution & Deployment

### 1. Running the Interactive Showroom (Streamlit)
To launch the visual dashboard locally to demonstrate the pipeline to clients:
```bash
pip install -r requirements-ui.txt
streamlit run streamlit_app.py
```

### 2. Local Production QA (FastAPI)
To boot the production API server locally via Uvicorn:
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```
*Interactive Swagger documentation will be live at:* `http://127.0.0.1:8000/docs`

### 3. Enterprise Docker Deployment
To package the API into an isolated, production-ready container image:
```bash
# Build the container
docker build -t realtime-ecommerce-intent-engine:v1 .

# Run and test the container locally
docker run -p 8000:8000 realtime-ecommerce-intent-engine:v1
```

---

## 📥 API Schema & Production Payload Example

The `/predict` endpoint expects a unified JSON payload. Behavioral tracking elements default to `0` or `0.0` to minimize initial payload size at early user landing phases.

**Sample Inbound Request (GTM Data Layer Payload):**
```json
{
  "VisitorType": "Returning_Visitor",
  "TrafficType": 3,
  "Browser": "Chrome",
  "OperatingSystems": "Mac",
  "Region": "Europe",
  "Month": "Nov",
  "Weekend": false,
  "Administrative": 0,
  "Informational": 0,
  "ProductRelated": 14,
  "ProductRelated_Duration": 342.5,
  "BounceRates": 0.0,
  "ExitRates": 0.012,
  "PageValues": 34.8
}
```

**Synchronous Response Packet:**
```json
{
  "conversion_probability": 0.8446,
  "high_intent_flag": true,
  "engine_used": "Closer Engine"
}
```
