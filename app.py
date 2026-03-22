from unicodedata import name

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import shap 

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="",
    layout="centered"
)

MELBOURNE_CBD = (-37.8136, 144.9631)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("melbourne_model.pkl")
    with open("model_features.json") as f:
        features = json.load(f)
    return model, features

model, feature_cols = load_model()

@st.cache_resource
def load_explainer(_model):
    # get the trained model inside pipeline
    regressor = _model.named_steps["regressor"]
    return shap.TreeExplainer(regressor)

explainer = load_explainer(model)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { max-width: 920px; margin: auto; }
        .result-box {
            background: #f0fdf4;
            border: 2px solid #22c55e;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-top: 16px;
        }
        .result-price {
            font-size: 2.4rem;
            font-weight: 700;
            color: #15803d;
        }
        .warning-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #78350f;
        }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Melbourne House Price Predictor")
st.caption("Enter a Melbourne address and property details to get an estimated sale price.")
st.divider()

# ── Address Input ─────────────────────────────────────────────────────────────
address = st.text_input(
    "Property Address",
    placeholder="e.g. 45 Collins St, Melbourne VIC 3000"
)

lat, lon, suburb, postcode, council, distance = None, None, "", "", "", None

if address:
    with st.spinner("Looking up address..."):
        try:
            geolocator = Nominatim(user_agent="melb_house_predictor_v1")
            time.sleep(1)                                                   # Nominatim rate limit
            location = geolocator.geocode(
                address + ", Victoria, Australia",
                addressdetails=True
            )

            if location:
                raw = location.raw.get("address", {})
                lat = location.latitude
                lon = location.longitude
                suburb = (
                    raw.get("suburb")
                    or raw.get("city_district")
                    or raw.get("town")
                    or raw.get("village")
                    or ""
                )
                postcode = str(raw.get("postcode", ""))
                council = raw.get("county", raw.get("state_district", ""))
                distance = round(geodesic(MELBOURNE_CBD, (lat, lon)).km, 2)

                st.success(f"Found: **{suburb}**, {postcode}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance from CBD", f"{distance} km")
                c2.metric("Latitude", round(lat, 4))
                c3.metric("Longitude", round(lon, 4))
                c4.metric("Postcode", postcode)
            else:
                st.error("Address not found. Try adding suburb and state, e.g. 'Richmond VIC'.")
        except Exception as e:
            st.error(f"Geocoding error: {e}")

# all we need from above geolocation is lat, lon, postcode, distance (from CBD)
# additionally we need user input for internal area, year built, plot size, rooms 

st.divider()

# ── Property Details ──────────────────────────────────────────────────────────
st.subheader("Property Details")
st.caption("Fill in the property details below. Location fields are auto-filled from the address.")

col1, col2 = st.columns(2)

with col1:
    prop_type = st.selectbox(
        "Property Type",
        options=["House", "Unit/Apartment", "Townhouse"],
        # format_func=lambda x: {"h": "House", "t": "Townhouse", "u": "Unit"}[x]
    )
    building_area = st.number_input("Building Area (m²)", min_value=10, max_value=1000, value=150)
    landsize = st.number_input("Plot Size (m²)", min_value=0, max_value=100000, value=500)

with col2:
    total_rooms = st.number_input(
        "Bed + Bath",
        min_value=1, max_value=20, value=5,
        # help="Bedrooms + bathrooms + other"
    )
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990)
    
    # propertycount = st.number_input(
    #     "Properties in Suburb (approx)",
    #     min_value=1, max_value=50000, value=5000,
    #     help="Rough number of properties in the suburb. Check realestate.com.au if unsure."
    # )
    propertycount = 5000

    suburb_input = st.text_input(
        "Suburb (auto-filled)",
        value=suburb,
        help="Auto-filled from address. Edit if incorrect."
    )
    # council_input = st.text_input(
    #     "Council Area (auto-filled)",
    #     value=council,
    #     help="Auto-filled from address. Edit if incorrect."
    # )
    council_input = ""

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

predict_ready = lat is not None or (suburb_input and postcode)

if st.button("Predict Price", type="primary", use_container_width=True, disabled=not predict_ready):
    if lat is None:
        st.warning("Please enter a valid address first.")
    else:
        try:
            input_df = pd.DataFrame([{
                "Suburb": suburb_input or suburb,
                "Type": prop_type,
                "Distance": distance,
                "Postcode": postcode,
                "Landsize": landsize,
                "BuildingArea": building_area,
                "YearBuilt": year_built,
                "CouncilArea": council_input or council,
                "Lattitude": lat,
                "Longtitude": lon,
                "Propertycount": propertycount,
                "Total_Internal_Rooms": total_rooms,
                #"Price_Category": "Medium"                  # placeholder
            }])

            prediction = np.expm1(model.predict(input_df)[0])

            # Transform input using preprocessor
            X_processed = model.named_steps["preprocessor"].transform(input_df)

            # Get feature names AFTER preprocessing
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            # Compute SHAP values
            shap_values = explainer.shap_values(X_processed)

           # Convert SHAP values from log space → price space
            prediction_log = model.predict(input_df)[0]
            prediction_price = np.expm1(prediction_log)

            shap_impacts = shap_values[0]

            # convert each feature impact to price contribution
            price_impacts = np.expm1(prediction_log + shap_impacts) - prediction_price

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "impact": price_impacts
            })

            # Sort by importance
            shap_df["abs_impact"] = shap_df["impact"].abs()
            shap_df = shap_df.sort_values(by="abs_impact", ascending=False).head(5)

            def clean_feature_name(name):
                return name.replace("num__", "").replace("cat__", "").replace("_", " ")

            shap_df["feature"] = shap_df["feature"].apply(clean_feature_name)

            st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:1rem; color:#166534; margin-bottom:6px;">Estimated Sale Price</div>
                    <div class="result-price">A${prediction:,.0f}</div>
                    <div style="font-size:0.85rem; color:#6b7280; margin-top:8px;">
                        Based on LightGBM model trained on Melbourne Housing Snapshot dataset
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### Why this price?")

            for _, row in shap_df.iterrows():
                sign = "+" if row["impact"] > 0 else "-"
                color = "#16a34a" if row["impact"] > 0 else "#dc2626"

                st.markdown(f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        padding:8px 12px;
                        margin-bottom:6px;
                        border-radius:6px;
                    ">
                        <span>{sign} {row['feature']}</span>
                        <span style="color:{color}; font-weight:600;">
                            {sign}${abs(row['impact']):,.0f}
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            # st.write("Input columns:", input_df.columns.tolist())
            # st.write("Model expects:", model.named_steps['preprocessor'].feature_names_in_)
            # st.write("Input columns:", input_df)

            # st.markdown("""
            #     <div class="warning-box" style="margin-top:16px;">
            #         <strong>Note:</strong> This is an estimate for educational purposes only.
            #         Actual property values depend on many factors not captured in this model.
            #     </div>
            # """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("This may happen if the suburb or council area wasn't seen during training. Try editing those fields.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LightGBM · scikit-learn · Streamlit · OpenStreetMap Nominatim")
