import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
from pydantic import BaseModel
from google.cloud import bigquery, storage

app = FastAPI(title="Realtime Ecommerce Intent Engine")

# --- CORS (Cross-Origin Ressource Sharing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you would put your exact Shopify URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------

# ... the rest of your GCP config and app.py code stays exactly the same ...

# --- GOOGLE CLOUD CONFIGURATION ---
# These will be read from environment variables in your Cloud Run setup
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gtm-m4299zzd-nti4m")
DATASET_ID = os.getenv("BQ_DATASET_ID", "ml_logs")
TABLE_ID = os.getenv("BQ_TABLE_ID", "intent_predictions_log")
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
BUCKET_NAME = f"intent-engine-models-{PROJECT_ID}" # <-- Your new globally unique bucket

# Initialize the GCP clients once at startup to reuse connection pools
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# --- DYNAMIC MODEL DECOUPLING (GCS) ---
def download_model_from_gcs(blob_name, destination_file_name):
    """Downloads a blob from the bucket and loads it into RAM."""
    print(f"Downloading {blob_name} from Cloud Storage...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    # Cloud Run instances have a writable /tmp/ directory in memory
    blob.download_to_filename(destination_file_name)
    return joblib.load(destination_file_name)

# --- COLD BOOT MODEL LOADING ---
print("Initializing Realtime Intent Engine...")
closer_model = download_model_from_gcs("conversion_engine_v1.joblib", "/tmp/conversion_engine_v1.joblib")
greeter_model = download_model_from_gcs("greeter_engine_v1.joblib", "/tmp/greeter_engine_v1.joblib")
print("✅ Models successfully loaded from Google Cloud Storage into RAM.")
OPTIMAL_THRESHOLD = 0.6010 


class CustomerJourneyInput(BaseModel):
    VisitorType: str
    TrafficType: int
    Browser: str
    OperatingSystems: str
    Region: str
    Month: str
    Weekend: bool
    Administrative: int
    Administrative_Duration: float  
    Informational: int
    Informational_Duration: float   
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float               

# --- BACKGROUND WORKER TASK ---
def log_to_bigquery(row_data: dict):
    """
    Executes a non-blocking streaming ingestion into BigQuery.
    This runs asynchronously AFTER the user has already received their response.
    """
    try:
        errors = bq_client.insert_rows_json(TABLE_REF, [row_data])
        if errors:
            print(f"BigQuery Ingestion Errors: {errors}")
    except Exception as e:
        print(f"Failed to stream telemetry log to BigQuery: {str(e)}")

# --- LIVE API ENDPOINT ---
@app.post("/predict")
async def predict_intent(data: CustomerJourneyInput, background_tasks: BackgroundTasks):
    
    # --- DATA TRANSLATION LAYER ---
    browser_map = {"Chrome": 2, "Safari": 1, "Firefox": 3, "Edge": 4}
    os_map = {"Windows": 1, "Mac": 2, "Linux": 3, "iOS": 4, "Android": 8}
    region_map = {"Europe": 1, "North America": 2, "Asia": 3, "South America": 4}

    input_data = {
        "VisitorType": data.VisitorType,
        "TrafficType": data.TrafficType,
        "Browser": browser_map.get(data.Browser, 2),  
        "OperatingSystems": os_map.get(data.OperatingSystems, 1),
        "Region": region_map.get(data.Region, 1),
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
    if data.ProductRelated == 0:
        greeter_features = ['VisitorType', 'TrafficType', 'Browser', 'OperatingSystems', 'Region', 'Month', 'Weekend']
        greeter_df = input_df[greeter_features]
        
        raw_probability = float(greeter_model.predict_proba(greeter_df)[0][1])
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