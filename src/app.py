import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the API
app = FastAPI(title="Conversion Prediction API - Dual Engine")

# 2. Define the Unified Pydantic Schema
# We give the behavioral metrics a default value of 0.0. 
# This way, the frontend only has to send the Day-One features at millisecond zero!
class ShopperPayload(BaseModel):
    # Day-One Features (The Greeter Needs These)
    VisitorType: str
    TrafficType: int
    Browser: int
    OperatingSystems: int
    Region: int
    Month: str
    Weekend: bool
    
    # Behavioral Features (The Closer Needs These) - Defaulting to Zero
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

# 3. Global variables for our two brains
greeter_model = None
closer_model = None

# 4. Load BOTH engines at startup
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

# 5. The Routing Endpoint
@app.post("/predict")
def predict_conversion(payload: ShopperPayload):
    # Convert incoming JSON into a single-row Pandas DataFrame
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # --- THE ROUTER LOGIC ---
    # If the user has viewed ANY pages, they are no longer at Millisecond Zero.
    has_browsed = (
        payload.ProductRelated > 0 or 
        payload.Administrative > 0 or 
        payload.Informational > 0
    )
    
    if not has_browsed:
        # Route to The Greeter
        engine_used = "Greeter Engine (Cold Start)"
        # The Greeter pipeline expects only its specific 7 features
        greeter_features = ['VisitorType', 'TrafficType', 'Browser', 'OperatingSystems', 'Region', 'Month', 'Weekend']
        model_input = input_df[greeter_features]
        probability = greeter_model.predict_proba(model_input)[0][1]
        
    else:
        # Route to The Closer
        engine_used = "Closer Engine (Engaged User)"
        # The Closer pipeline expects the full behavioral dataset
        probability = closer_model.predict_proba(input_df)[0][1]
        
    # Convert math to business logic
    is_high_intent = bool(probability > 0.5)
    
    # Return the prediction AND the diagnostic router info
    return {
        "engine_used": engine_used,
        "conversion_probability": round(probability, 4),
        "high_intent_flag": is_high_intent
    }
