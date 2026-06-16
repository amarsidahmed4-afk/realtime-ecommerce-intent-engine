import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from google.cloud import bigquery

app = FastAPI(title="Realtime Ecommerce Intent Engine")

# --- BIGQUERY CONFIGURATION ---
# These will be read from environment variables in your Cloud Run setup
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gtm-m4299zzd-nti4m")
DATASET_ID = os.getenv("BQ_DATASET_ID", "ecommerce_telemetry")
TABLE_ID = os.getenv("BQ_TABLE_ID", "intent_predictions_log")
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize the BQ client once at startup to reuse connection pools
bq_client = bigquery.Client(project=PROJECT_ID)

# --- COLD BOOT MODEL LOADING ---
closer_model = joblib.load("models/conversion_engine_v1.joblib")
greeter_model = joblib.load("models/greeter_engine_v1.joblib")
OPTIMAL_THRESHOLD = 0.70

class CustomerJourneyInput(BaseModel):
    VisitorType: str
    TrafficType: int
    Browser: str
    OperatingSystems: str
    Region: str
    Month: str
    Weekend: bool
    Administrative: int
    Administrative_Duration: float  # Ensure these duration tracking variables exist
    Informational: int
    Informational_Duration: float   # Ensure these duration tracking variables exist
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float               # Ensure this traffic-weight variable exists

# --- BACKGROUND WORKER TASK ---
def log_to_bigquery(row_data: dict):
    """
    Executes a non-blocking streaming ingestion into BigQuery.
    This runs asynchronously AFTER the user has already received their response.
    """
    try:
        # insert_rows_json expects a list of dictionaries (rows)
        errors = bq_client.insert_rows_json(TABLE_REF, [row_data])
        if errors:
            # We log to stdout/stderr so Cloud Logging captures it without crashing the app
            print(f"BigQuery Ingestion Errors: {errors}")
    except Exception as e:
        print(f"Failed to stream telemetry log to BigQuery: {str(e)}")

# --- LIVE API ENDPOINT ---
@app.post("/predict")
async def predict_intent(data: CustomerJourneyInput, background_tasks: BackgroundTasks):
    
    # --- FIX: Convert the incoming Pydantic schema completely to a 1-row DataFrame ---
    # This ensures all categorical and numerical features exist with exact column names
    input_data = {
        "VisitorType": data.VisitorType,
        "TrafficType": data.TrafficType,
        "Browser": data.Browser,
        "OperatingSystems": data.OperatingSystems,
        "Region": data.Region,
        "Month": data.Month,
        "Weekend": data.Weekend,
        "Administrative": data.Administrative,
        "Administrative_Duration": data.Administrative_Duration,
        "Informational": data.Informational,
        "Informational_Duration": data.Informational_Duration,
        "ProductRelated": data.ProductRelated,
        "ProductRelated_Duration": data.ProductRelated_Duration,
        "BounceRates": data.BounceRates,
        "ExitRates": data.ExitRates,
        "PageValues": data.PageValues,
        "SpecialDay": data.SpecialDay
    }
    
    input_df = pd.DataFrame([input_data])
    
    # --- DYNAMIC ROUTING MATRIX ---
    # The rest of your logic remains exactly the same!
    if data.ProductRelated == 0:
        raw_probability = float(greeter_model.predict_proba(input_df)[0][1])
        engine_tag = "Greeter Engine"
    else:
        raw_probability = float(closer_model.predict_proba(input_df)[0][1])
        engine_tag = "Closer Engine"
        
    # 3. Apply F1-Optimized Business Logic
    high_intent_flag = bool(raw_probability >= OPTIMAL_THRESHOLD)
    
    # 4. Construct the Permanent Telemetry Record
    telemetry_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "visitor_type": data.VisitorType,
        "browser": data.Browser,
        "operating_system": data.OperatingSystems,
        "product_related_pages": data.ProductRelated,
        "page_values": data.PageValues,
        "conversion_probability": raw_probability,
        "high_intent_flag": high_intent_flag,
        "engine_used": engine_tag
    }
    
    # 5. Hand the logging job off to the background thread pool
    background_tasks.add_task(log_to_bigquery, telemetry_log)
    
    # 6. Return instantly to GTM (Track A)
    return {
        "conversion_probability": raw_probability,
        "high_intent_flag": high_intent_flag,
        "engine_used": engine_tag
    }