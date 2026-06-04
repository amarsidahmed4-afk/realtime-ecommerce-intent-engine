from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the Web App
app = FastAPI(
    title="Marketing Conversion Engine API",
    description="Live scoring API for predicting online shopper conversion.",
    version="1.0"
)

# 2. Load the Engine (Happens once when the server boots)
try:
    print("Loading ML Pipeline...")
    model = joblib.load('models/conversion_engine_v1.joblib')
    print("Pipeline loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the Data Schema (The "Bouncer")
# We set logical default values so we don't have to type all 17 fields every time we test
class VisitorData(BaseModel):
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
    Month: str = "May"
    OperatingSystems: int = 1
    Browser: int = 1
    Region: int = 1
    TrafficType: int = 1
    VisitorType: str = "Returning_Visitor"
    Weekend: bool = False

# 4. Define the API Endpoint
@app.post("/predict")
def predict_conversion(visitor: VisitorData):
    try:
        # Convert the incoming JSON into a pandas DataFrame row
        visitor_dict = visitor.model_dump()
        input_df = pd.DataFrame([visitor_dict])
        
        # Run the full preprocessing & prediction pipeline
        prob = model.predict_proba(input_df)[:, 1][0]
        
        # Return the clean JSON response
        return {
            "status": "success",
            "conversion_probability": round(float(prob), 4),
            "high_intent_flag": bool(prob > 0.70) # Using our custom threshold!
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))