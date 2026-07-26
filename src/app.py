import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery, storage

app = FastAPI(title="Realtime Ecommerce Intent Engine")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update to specific Shopify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gtm-m4299zzd-nti4m")
DATASET_ID = os.getenv("BQ_DATASET_ID", "ml_logs")
TABLE_ID = os.getenv("BQ_TABLE_ID", "intent_predictions_log")
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
BUCKET_NAME = f"intent-engine-models-{PROJECT_ID}"
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", "0.6010"))

# Initialize GCP clients
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)
except Exception as e:
    print(f"⚠️ GCP Client initialization warning: {e}")
    bq_client, storage_client = None, None

def load_model(blob_name: str, local_path: str):
    """Downloads from GCS if available; falls back to local file for offline dev."""
    try:
        if storage_client and os.getenv("GAE_ENV", os.getenv("K_SERVICE")): # In Cloud Run
            print(f"Downloading {blob_name} from Cloud Storage...")
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            return joblib.load(local_path)
    except Exception as e:
        print(f"⚠️ Could not load from GCS: {e}. Attempting local load...")

    # Fallback to local models directory
    fallback_paths = [f"../models/{blob_name}", f"models/{blob_name}", local_path]
    for path in fallback_paths:
        if os.path.exists(path):
            print(f"✅ Loaded model locally from: {path}")
            return joblib.load(path)
            
    raise RuntimeError(f"❌ Critical Error: Could not locate model artifact {blob_name}")

# Cold Boot Model Loading
print("Initializing Realtime Intent Engine...")
closer_model = load_model("conversion_engine_v1.joblib", "/tmp/conversion_engine_v1.joblib")
greeter_model = load_model("greeter_engine_v1.joblib", "/tmp/greeter_engine_v1.joblib")
print("✅ Engines successfully loaded into RAM.")