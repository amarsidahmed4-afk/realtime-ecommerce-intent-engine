# ⚡ Realtime Ecommerce Intent Engine

A high-performance, context-aware machine learning microservice built to optimize real-time e-commerce conversion pipelines. This service intercepts live web traffic telemetry via Google Tag Manager (GTM), analyzes the behavioral context of the customer journey, and dynamically routes payloads to specialized machine learning engines to predict purchase intent in milliseconds.

## 🏗️ The 5-Pillar Architecture

The infrastructure is designed as a decoupled, serverless ecosystem built for infinite scale and zero idle-cost compute:

1. **Ingestion (Google Tag Manager):** Captures high-intent customer behavior on the frontend via a low-latency Data Layer without touching the core website codebase.
2. **Infrastructure (Docker & Google Cloud Run):** Containerized using Docker to eliminate environment conflict, deployed as a stateless microservice that auto-scales to handle massive concurrent traffic spikes.
3. **The ML Brain (FastAPI & LightGBM):** Runs an asynchronous FastAPI web server that passes incoming web traffic through custom Target Encoders, executing rapid-fire inferences via specialized models.
4. **Observability (BigQuery):** Logs every single inference call in real-time to a permanent Google BigQuery data warehouse for auditability and drift detection via non-blocking background tasks.
5. **Business & ROI Auditing:** Automated pipelines query the live BigQuery ledger against actual client checkouts to validate real-world drift and calculate exact financial ROI.

---

## 🔀 Dynamic Routing Matrix

To maximize performance across different stages of the online customer journey, the API executes a custom routing matrix across two pre-trained `.joblib` pipelines:

* **The Greeter Engine (Top of Funnel):** Activates at "Millisecond Zero" when a user first lands on the site (0 product page views). It relies exclusively on day-one categorical data using pre-fitted Target Encoders with Bayesian smoothing to predict baseline conversion propensity.
* **The Closer Engine (Bottom of Funnel):** Automatically takes over the moment a user views a single product page. It evaluates deep session engagement metrics (Page Values, Exit Rates, durations) using a heavily optimized LightGBM classifier.

> 📊 **Business Logic Optimization:** Rather than relying on standard, unoptimized default decision boundaries (50%), this engine applies a strict, hardcoded threshold of **70%**—discovered via a 200-trial Bayesian optimization sweep—maximizing the F1-Score to shield businesses from wasting marketing spend on false positives.

---

## 🗂️ Repository Structure

```text
├── .github/workflows/    # CI/CD automated deployment pipelines
├── models/               # Serialized mathematical brains (.joblib)
├── notebooks/            # Phase 1: EDA, Optimization & Feature Engineering
│   ├── utils/            # Custom visualization and statistical evaluation scripts
│   ├── 01_exploratory.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_cold_start_model.ipynb
├── src/                  # The Production Factory Floor
│   ├── app.py            # FastAPI Application & Background Routing
│   ├── marketing_pipeline.py    # Target Encoding & Data Translation Layer
│   └── production_audit.py      # Standalone BigQuery ROI & Drift Validator
├── Dockerfile            # Containerization recipe for Google Cloud Run
├── GTM_INTEGRATION.md    # Frontend client onboarding documentation
├── requirements.txt      # Production backend dependencies
└── requirements-dev.txt  # Local data science lab packages
```

---

## 🚀 Execution & Deployment

### 1. Enterprise Docker Deployment
To package the API into an isolated, production-ready container image:
```bash
# Build the container
docker build -t realtime-ecommerce-intent-engine:v1 .

# Run and test the container locally
docker run -p 8000:8000 realtime-ecommerce-intent-engine:v1
```

### 2. Local Production QA (FastAPI)
To boot the production API server locally via Uvicorn:
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```
*Interactive Swagger documentation will be live at:* `http://127.0.0.1:8000/docs`

### 3. Client Onboarding
To integrate a new e-commerce storefront with this engine, please refer directly to the `GTM_INTEGRATION.md` handoff guide for the standardized JavaScript dataLayer schema.

---

## 📥 API Schema & Production Payload Example

The `/predict` endpoint expects a unified JSON payload. Behavioral tracking elements default to `0` or `0.0` to minimize initial payload size at early user landing phases.

**Synchronous Response Packet:**
```json
{
  "conversion_probability": 0.8446,
  "high_intent_flag": true,
  "engine_used": "Closer Engine"
}
```