# 🚀 Realtime E-Commerce Intent Engine: Release Protocol

**WARNING:** This is a decoupled serverless architecture. Changes to one system (ML, API, or Tracking) can break upstream/downstream components if not deployed in the correct sequence. 

Always follow the strict order of operations below.

---

## 🧠 PHASE 1: ML Model Upgrades (The Brains)
*Models are strictly decoupled from the API. NEVER commit `.joblib` files to GitHub.*

1. Run hyperparameter sweeps in `src/marketing_evaluation.py`.
2. Generate the optimized `conversion_engine_vX.joblib` locally.
3. Push the new model directly to the secure Google Cloud Storage bucket:
```bash
gsutil cp models/*.joblib gs://intent-engine-models-[YOUR_PROJECT_ID]/
```
4. Verify the file exists and the timestamp is updated in the GCP Storage Console.

---

## ⚙️ PHASE 2: API Upgrades (The Body)
*The FastAPI app translates GTM payloads and fetches the ML models from GCS.*

1. Update `src/app.py` locally. 
   - *CRITICAL:* If you changed the required inputs in your LightGBM model, you MUST update the `CustomerJourneyInput` Pydantic class to match.
   - *CRITICAL:* Increment the `"api_version": "vX.X"` in the final JSON return payload.
2. Commit and push the code to GitHub:
```bash
git add .
git commit -m "feat: upgrade intent engine to v2.0"
git push origin main
```
3. **Wait for GitHub Actions:** Monitor the Actions tab in GitHub. Wait for the green checkmark indicating successful deployment to Cloud Run.
4. **Cold Boot Verification:** Open the Cloud Run Logs in GCP. Filter by "Error" and verify the container started successfully and printed: `✅ Models successfully loaded from Google Cloud Storage into RAM.`

---

## 🕸️ PHASE 3: Tagging Upgrades (The Nervous System)
*Google Tag Manager must precisely match the API's Pydantic schema.*

1. Open **GTM Server-Side Container**.
2. If `app.py` requires new fields, open the `HTTP - Forward Context to FastAPI` tag and add the new variables to the **Request Body**.
   - *Ensure flat JSON structure unless otherwise specified.*
3. Verify the **Destination URL** still points to the active Cloud Run service URL.
4. Click **Preview**, trigger a test click on the live store, and verify a **200 OK** status from the API.
5. Hit **Submit** -> **Publish** in GTM Server-Side.
6. Hit **Submit** -> **Publish** in GTM Web Container (if State Machine JS was modified).

---

## 🚨 EMERGENCY ROLLBACK PROTOCOL
If a deployment causes a spike in 500 or 422 errors:
1. Go to **Google Cloud Run** -> select the service -> click **Revisions**.
2. Select the previous stable revision and click **Migrate Traffic (100%)**.
3. Go to **Google Tag Manager Server-Side** -> click **Versions**.
4. Select the previous stable version -> click **Set as Latest Version** -> **Publish**.
