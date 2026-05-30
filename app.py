import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Stacking Regression",
    layout="wide"
)

# ------------------------------------------------
# Load CSS
# ------------------------------------------------
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model, Scaler & Results
# ------------------------------------------------
model   = pickle.load(open("models/Stacking_Regressor.pkl", "rb"))
scaler  = pickle.load(open("models/scaler.pkl", "rb"))
results = pickle.load(open("models/r2_results.pkl", "rb"))

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown(
    "<h1 class='main-title'>House Price Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='info-box'>
    Predict the <b>House Price</b> based on property features
    using <b>Stacking Ensemble Regression</b> 
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# Sidebar – Model Information
# ------------------------------------------------
st.sidebar.header("⚙️ Model Information")

st.sidebar.success("Algorithm : Stacking Regressor")
st.sidebar.info("Base Learners:")
st.sidebar.write("• Decision Tree Regressor")
st.sidebar.write("• Random Forest Regressor")
st.sidebar.write("• Gradient Boosting Regressor")
st.sidebar.write("• K-Nearest Neighbors Regressor")
st.sidebar.info("Meta Learner:")
st.sidebar.write("• Ridge Regression")
st.sidebar.success("Dataset : Housing Price Prediction")

# ------------------------------------------------
# Performance Section
# ------------------------------------------------
st.markdown("---")
st.subheader("Model Performance Comparison (R² Score)")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

model_labels = {
    "decision_tree":  "Decision Tree",
    "random_forest":  "Random Forest",
    "gradient_boost": "Gradient Boost",
    "knn":            "KNN",
    "stacking":       "🏆 Stacking"
}

for col, (key, label) in zip(
    [col_m1, col_m2, col_m3, col_m4, col_m5],
    model_labels.items()
):
    with col:
        r2_val  = results.get(key, {}).get("R2", 0)
        mae_val = results.get(key, {}).get("MAE", 0)
        st.metric(label=label, value=f"{r2_val}%", delta=f"MAE: ${mae_val:,.0f}")

# ------------------------------------------------
# Encoding Dictionaries
# ------------------------------------------------
location_dict = {
    "Rural":    0,
    "Suburban": 2,
    "Urban":    3
}

condition_dict = {
    "Average":   0,
    "Excellent": 1,
    "Good":      2,
    "Poor":      3
}

# ------------------------------------------------
# User Inputs
# ------------------------------------------------
st.markdown("---")
st.subheader("🧾 Enter Property Features")

col1, col2, col3 = st.columns(3)

with col1:
    area      = st.number_input("Area (sq ft)", min_value=500,  max_value=5000, value=1500, step=50)
    bedrooms  = st.slider("Bedrooms",  min_value=1, max_value=5, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=3, value=2)

with col2:
    age      = st.slider("Age of Property (years)", min_value=0, max_value=50, value=10)
    garage   = st.slider("Garage Spaces", min_value=0, max_value=2, value=1)

with col3:
    location  = st.selectbox("Location",  list(location_dict.keys()))
    condition = st.selectbox("Condition", list(condition_dict.keys()))

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button(" Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        age,
        garage,
        location_dict[location],
        condition_dict[condition]
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]

    # Base learner predictions
    base_preds = {}
    base_names = ["Decision Tree", "Random Forest", "Gradient Boost", "KNN"]
    for i, bname in enumerate(base_names):
        bp = model.estimators_[i].predict(input_scaled)[0]
        base_preds[bname] = bp

    st.markdown(
        f"""
        <div class='prediction-box'>
            Estimated House Price
            <br><br>
            <span>${prediction:,.0f}</span>
            <br><br>
            Based on Stacking Ensemble of 4 Base Learners + Ridge Meta Learner
        </div>
        """,
        unsafe_allow_html=True
    )

    # Individual base learner predictions
    st.markdown("---")
    st.subheader(" Base Learner Price Estimates")

    vote_cols = st.columns(4)
    for col, (bname, bprice) in zip(vote_cols, base_preds.items()):
        with col:
            st.info(f"**{bname}**\n\n${bprice:,.0f}")

# ------------------------------------------------
# Feature Importance Info
# ------------------------------------------------
st.markdown("---")
st.subheader(" Feature Descriptions")

with st.expander("Area"):
    st.write("Total area of the house in square feet (500 – 5000 sq ft)")

with st.expander("Bedrooms & Bathrooms"):
    st.write("Number of bedrooms (1–5) and bathrooms (1–3)")

with st.expander("Age of Property"):
    st.write("How old the property is in years (0 = brand new, 50 = oldest)")

with st.expander("Garage"):
    st.write("Number of garage parking spaces (0, 1, or 2)")

with st.expander("Location"):
    st.write("""
    • Urban    – City area, highest demand  
    • Suburban – Outskirts, moderate demand  
    • Rural    – Village/remote area
    """)

with st.expander("Condition"):
    st.write("""
    • Excellent – Newly renovated, premium finish  
    • Good      – Well maintained  
    • Average   – Standard condition  
    • Poor      – Needs repairs
    """)

# ------------------------------------------------
# How Stacking Works
# ------------------------------------------------
st.markdown("---")
st.subheader(" How Stacking Regression Works")

st.markdown("""
<div class='info-box'>
<b>Level 0 — Base Learners</b> : Decision Tree, Random Forest, Gradient Boosting, KNN<br>
Each base learner is trained independently and produces price predictions as <i>meta-features</i>.<br><br>
<b>Level 1 — Meta Learner</b> : Ridge Regression<br>
Ridge Regression is trained on the base learner predictions to produce the final refined price estimate.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown(
    """
    <hr>
    <center>
    <h4> Built using Streamlit, Scikit-Learn & Stacking Ensemble </h4>
    </center>
    """,
    unsafe_allow_html=True
)
