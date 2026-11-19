"""
Smart Waste Category Classifier - Interactive Web App (Streamlit)
Author: Sheikh Kamran Yasir Hussain (ID: 123626 | Batch: B1 | 2026)
Run: streamlit run src/app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smart Waste Category Classifier",
    page_icon="♻️",
    layout="centered",
)

BIN_GUIDE = {
    "Dry/Recyclable": {
        "color": "🔵 Blue Bin",
        "tip": "Rinse and keep dry before placing in the recycling bin.",
    },
    "Wet/Organic": {
        "color": "🟢 Green Bin",
        "tip": "Ideal for composting and biogas generation.",
    },
    "E-Waste": {
        "color": "🟣 Purple / Certified E-Waste Drop-Off",
        "tip": "Never mix with household trash; take to an authorized e-waste collection center.",
    },
    "Hazardous": {
        "color": "🔴 Red / Special Hazardous Bin",
        "tip": "Handle carefully; keep sealed and dispose via municipal hazardous waste collection.",
    },
    "General/Landfill": {
        "color": "⚫ Black / Grey Bin",
        "tip": "Non-recyclable residual waste destined for sanitary landfill.",
    },
}


@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "waste_classifier_dt.pkl")
    return joblib.load(model_path)


st.title("♻️ Smart Waste Category Classifier")
st.caption("AI-Based Project | Sheikh Kamran Yasir Hussain (ID: 123626 • Batch: B1 • 2026)")
st.markdown(
    "Enter the properties of any waste item below to get an instant **Decision Tree** disposal recommendation."
)

model = load_model()

with st.form("waste_form"):
    col1, col2 = st.columns(2)
    with col1:
        item_type = st.text_input("Item Name", value="PET Water Bottle", placeholder="e.g., Banana Peel, USB Cable")
        material = st.selectbox(
            "Primary Material",
            [
                "Plastic",
                "Metal",
                "Paper",
                "Cardboard",
                "Glass",
                "Organic",
                "Electronic",
                "Chemical",
                "Multi-layer Plastic",
                "Ceramic",
            ],
        )
    with col2:
        weight_grams = st.number_input("Approximate Weight (grams)", min_value=1, max_value=5000, value=25)
        is_recyclable = st.radio("Is the item marked/known recyclable?", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("Classify Waste Item", use_container_width=True)

if submitted:
    sample = pd.DataFrame(
        [
            {
                "item_type": item_type.strip(),
                "material": material,
                "weight_grams": weight_grams,
                "is_recyclable": is_recyclable,
            }
        ]
    )
    prediction = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]
    confidence = max(probs) * 100

    info = BIN_GUIDE.get(prediction, {"color": "Standard Bin", "tip": ""})

    st.success(f"### Recommended Category: **{prediction}**")
    st.info(f"**Bin Guide:** {info['color']} — {info['tip']}")
    st.metric(label="Model Prediction Confidence", value=f"{confidence:.1f}%")

    prob_df = pd.DataFrame(
        {"Waste Category": model.classes_, "Probability (%)": [round(p * 100, 2) for p in probs]}
    ).sort_values(by="Probability (%)", ascending=False)
    st.bar_chart(prob_df.set_index("Waste Category"))
