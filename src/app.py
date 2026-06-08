import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import bigquery
from datetime import datetime, timezone

# Initialize BigQuery Client
bq_client = bigquery.Client(project="gtm-m4299zzd-nti4m")
TABLE_ID = "gtm-m4299zzd-nti4m.ml_logs.predictions"

# --- 1. MAPPING DICTIONARIES ---
# Translating GTM strings back to the dataset's numerical encoding format.
# If a strange device hits the site, we default to 1 (most common category) to prevent crashes.
OS_MAP = {"Windows": 1, "Mac": 2, "Linux": 3, "Chrome OS": 4, "iOS": 5, "Android": 6}
BROWSER_MAP = {"Chrome": 1, "Safari": 2, "Edge": 3, "Firefox": 4, "Opera": 5}
REGION_MAP = {"North America": 1, "Europe": 2, "Asia": 3, "South America": 4, "Other": 5}

# 2. Initialize the API
app = FastAPI(title="Conversion Prediction API - Dual Engine")

# 3. Define the Unified Pydantic Schema
class ShopperPayload(BaseModel):
    # Day-One Features
    VisitorType: str
    TrafficType: int
    Browser: str             # <-- Updated to str for GTM
    OperatingSystems: str    # <-- Updated to str for GTM
    Region: str              # <-- Updated to str for GTM
    Month: str
    Weekend: bool
    
    # Behavioral Features - Defaulting to Zero
    Administrative: int = 0
    Administrative_Duration: float = 0.0
    Informational: int = 0
    Informational_Duration: float = 0.0
    ProductRelated: int = 0
    ProductRelated_Duration: float = 0.0
    BounceRates: float = 0.0
    ExitRates: float = 0.0
    PageValues: float = 0.0
    SpecialDay: float = 0.0

# 4. Global variables for our two brains
greeter_model = None
closer_model = None

# 5. Load BOTH engines at startup
@app.on_event("startup")
def load_models():
    global greeter_model, closer_model
    try:
        print("Loading The Greeter (Top of Funnel)...")
        greeter_model = joblib.load("models/greeter_engine_v1.joblib")
        
        print("Loading The Closer (Bottom of Funnel)...")
        closer_model = joblib.load("models/conversion_engine_v1.joblib")
        
        print("Both ML Engines loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")

# 6. The Routing Endpoint
@app.post("/predict")
def predict_conversion(payload: ShopperPayload):
    input_dict = payload.model_dump()
    
    # --- PRE-PROCESSING THE STRINGS ---
    # Intercept the GTM strings and swap them for the ML integers
    input_dict['Browser'] = BROWSER_MAP.get(input_dict['Browser'], 1)
    input_dict['OperatingSystems'] = OS_MAP.get(input_dict['OperatingSystems'], 1)
    input_dict['Region'] = REGION_MAP.get(input_dict['Region'], 1)

    # Now build the DataFrame safely
    input_df = pd.DataFrame([input_dict])
    
    # --- THE ROUTER LOGIC ---
    has_browsed = (
        payload.ProductRelated > 0 or 
        payload.Administrative > 0 or 
        payload.Informational > 0
    )
    
    if not has_browsed:
        engine_used = "Greeter Engine (Cold Start)"
        greeter_features = ['VisitorType', 'TrafficType', 'Browser', 'OperatingSystems', 'Region', 'Month', 'Weekend']
        model_input = input_df[greeter_features]
        probability = greeter_model.predict_proba(model_input)[0][1]
        
    else:
        engine_used = "Closer Engine (Engaged User)"
        probability = closer_model.predict_proba(input_df)[0][1]
        
    # Convert math to business logic
    is_high_intent = bool(probability > 0.5)
    
    # Define the final payload
    response_data = {
        "engine_used": engine_used,
        "conversion_probability": round(probability, 4),
        "high_intent_flag": is_high_intent
    }
    
    # --- OBSERVABILITY: Shout to the Cloud Run Logs ---
    print("\n" + "="*40)
    print("🚀 NEW SHOPPER INGESTED VIA GTM")
    print(f"🌍 Demographics: Browser={payload.Browser}, OS={payload.OperatingSystems}, Region={payload.Region}")
    print(f"🧠 ML Routing: {engine_used}")
    print(f"🎯 Conversion Probability: {round(probability * 100, 2)}%")
    print(f"🔥 High Intent: {is_high_intent}")
    print("="*40 + "\n")

    # --- BIGQUERY LOGGING ---
    try:
        row_to_insert = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "visitor_type": payload.VisitorType,
            "engine_used": engine_used,
            "conversion_probability": float(probability),
            "high_intent_flag": is_high_intent
        }]
        
        # Stream the data into the table
        errors = bq_client.insert_rows_json(TABLE_ID, row_to_insert)
        if errors:
            print(f"⚠️ BigQuery Insert Errors: {errors}")
        else:
            print("💾 Successfully logged to BigQuery")
            
    except Exception as e:
        print(f"⚠️ Failed to communicate with BigQuery: {e}")
        
    return response_data