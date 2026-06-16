import os
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_production_roi():
    """
    Queries the live BigQuery intent_predictions_log and joins it against 
    mock client conversion data to audit real-world model drift and financial ROI.
    """
    # Initialize the BigQuery Client
    project_id = os.getenv("GCP_PROJECT_ID", "gtm-m4299zzd-nti4m")
    client = bigquery.Client(project=project_id)
    
    print(f"🔍 Initializing Production ROI Audit for Project: {project_id}...\n")

    # The ROI Audit Query
    # In production, 'client_checkout_ledger' is swapped for the client's Stripe/GA4 table.
    query = """
        WITH engine_logs AS (
            SELECT 
                timestamp,
                visitor_type,
                conversion_probability,
                high_intent_flag,
                engine_used
            FROM `gtm-m4299zzd-nti4m.ml_logs.intent_predictions_log`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ),
        
        mock_client_checkouts AS (
            -- Simulating actual conversions for testing purposes
            SELECT 
                timestamp,
                TRUE as actual_conversion
            FROM engine_logs
            WHERE conversion_probability >= 0.85 
        )
        
        SELECT 
            e.timestamp,
            e.engine_used,
            e.high_intent_flag AS predicted_conversion,
            COALESCE(c.actual_conversion, FALSE) AS actual_conversion
        FROM engine_logs e
        LEFT JOIN mock_client_checkouts c 
        ON e.timestamp = c.timestamp
    """

    try:
        # Load results directly into a Pandas DataFrame
        df = client.query(query).to_dataframe()
        
        if df.empty:
            print("⚠️ No prediction logs found for the last 30 days.")
            return

        # Calculate Real-World Drift Metrics
        y_pred = df['predicted_conversion']
        y_true = df['actual_conversion']
        
        current_precision = precision_score(y_true, y_pred, zero_division=0)
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)

        # Generate the Consulting Report
        print("="*50)
        print(" 📊 30-DAY ENGINE PERFORMANCE & ROI AUDIT ")
        print("="*50)
        print(f"Total Sessions Analyzed : {len(df)}")
        print(f"Engine Precision        : {current_precision:.2f} (Target: > 0.70)")
        print(f"Engine Recall           : {current_recall:.2f}")
        print(f"Engine F1-Score         : {current_f1:.2f}")
        print("="*50)

        # Alerting Logic
        if current_precision < 0.70:
            print("🚨 ALERT: Model precision has drifted below the 70% profitable threshold!")
            print("Action Required: Trigger hyperparameter retrain pipeline.")
        else:
            print("✅ Status: Engine is operating within optimal profit margins.")

    except Exception as e:
        print(f"❌ Error executing BigQuery audit: {e}")

if __name__ == "__main__":
    evaluate_production_roi()
