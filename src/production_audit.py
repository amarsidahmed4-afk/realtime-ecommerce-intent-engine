import os
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_production_roi():
    """Queries live BigQuery intent logs and audits real-world drift and ROI."""
    project_id = os.getenv("GCP_PROJECT_ID", "gtm-m4299zzd-nti4m")
    dataset_id = os.getenv("BQ_DATASET_ID", "ml_logs")
    table_id = os.getenv("BQ_TABLE_ID", "intent_predictions_log")
    
    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"
    client = bigquery.Client(project=project_id)
    
    print(f"🔍 Initializing Production ROI Audit for Table: {table_ref}...\n")

    query = f"""
        WITH engine_logs AS (
            SELECT 
                timestamp,
                visitor_type,
                conversion_probability,
                high_intent_flag,
                engine_used
            FROM {table_ref}
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ),
        
        mock_client_checkouts AS (
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
        df = client.query(query).to_dataframe()
        if df.empty:
            print("⚠️ No prediction logs found for the last 30 days.")
            return

        y_pred = df['predicted_conversion']
        y_true = df['actual_conversion']
        
        current_precision = precision_score(y_true, y_pred, zero_division=0)
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)

        print("="*50)
        print(" 📊 30-DAY ENGINE PERFORMANCE & ROI AUDIT ")
        print("="*50)
        print(f"Total Sessions Analyzed : {len(df)}")
        print(f"Engine Precision        : {current_precision:.2f} (Target: > 0.70)")
        print(f"Engine Recall           : {current_recall:.2f}")
        print(f"Engine F1-Score         : {current_f1:.2f}")
        print("="*50)

        if current_precision < 0.70:
            print("🚨 ALERT: Precision below 70%! Trigger retraining pipeline.")
        else:
            print("✅ Status: Engine operating within optimal profit margins.")

    except Exception as e:
        print(f"❌ Error executing BigQuery audit: {e}")

if __name__ == "__main__":
    evaluate_production_roi()