import streamlit as st
import requests

# --- CONFIGURATION ---
API_URL = "https://conversion-api-service-347039794179.europe-west1.run.app/predict"

st.set_page_config(page_title="Conversion AI Pitch Simulator", layout="wide")

st.title("🎯 Live Customer Intent & Conversion Simulator")
st.markdown("Adjust the incoming traffic signals below to see how the dual-engine ML architecture routes the user and predicts conversion probability in real-time.")

# --- UI LAYOUT ---
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.header("🌍 Day-One Signals")
    st.markdown("*(Top of Funnel / Cold Start)*")
    visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])
    browser = st.selectbox("Browser", ["Chrome", "Safari", "Edge", "Firefox", "Opera"])
    os = st.selectbox("Operating System", ["Windows", "Mac", "iOS", "Android", "Linux"])
    region = st.selectbox("Region", ["Europe", "North America", "Asia", "South America", "Africa"])
    month = st.selectbox("Month", ["May", "Nov", "Mar", "Dec", "Oct", "Sep", "Aug", "Jul", "June", "Feb"])
    traffic_type = st.slider("Traffic Source Type", 1, 20, 2)
    weekend = st.toggle("Is Weekend?", value=False)

with col2:
    st.header("🖱️ Session Behavior")
    st.markdown("*(Bottom of Funnel / Engaged)*")
    st.info("Adding Pageviews triggers the Closer Engine.")
    product_related = st.slider("Product Pages Viewed", 0, 50, 0)
    product_duration = st.slider("Product View Duration (sec)", 0.0, 1000.0, 0.0)
    page_values = st.slider("Page Values (Cart/Checkout Hits)", 0.0, 100.0, 0.0)
    
    with st.expander("Advanced Metrics"):
        admin_pages = st.number_input("Admin Pages", 0, 20, 0)
        info_pages = st.number_input("Info Pages", 0, 20, 0)
        bounce_rates = st.slider("Bounce Rate", 0.0, 0.2, 0.0)
        exit_rates = st.slider("Exit Rate", 0.0, 0.2, 0.0)

with col3:
    st.header("🧠 Live ML Inference")
    st.markdown("---")
    
    # --- BUILD THE PAYLOAD ---
    payload = {
        "VisitorType": visitor_type,
        "TrafficType": traffic_type,
        "Browser": browser,
        "OperatingSystems": os,
        "Region": region,
        "Month": month,
        "Weekend": weekend,
        "Administrative": admin_pages,
        "Administrative_Duration": 0.0,
        "Informational": info_pages,
        "Informational_Duration": 0.0,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": 0.0
    }

    if st.button("🚀 Calculate Conversion Intent", use_container_width=True):
        with st.spinner("Pinging API in Belgium..."):
            try:
                # Send to Cloud Run
                response = requests.post(API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    prob = data["conversion_probability"]
                    engine = data["engine_used"]
                    
                    # Display Results
                    st.success("Analysis Complete!")
                    st.metric(label="Routing Engine Activated", value=engine)
                    
                    # Visual Probability Bar
                    st.subheader(f"Conversion Probability: {prob * 100:.1f}%")
                    st.progress(prob)
                    
                    if data["high_intent_flag"]:
                        st.error("🔥 HIGH INTENT USER DETECTED - TRIGGER INCENTIVE TAGS")
                    else:
                        st.warning("🧊 COLD USER - DO NOT WASTE AD SPEND")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")