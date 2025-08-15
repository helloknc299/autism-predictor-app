import streamlit as st
import pandas as pd
import pickle

# ------------------ Load model ------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ Load encoders ------------------
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res',
                       'used_app_before', 'relation']

feature_order = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'contry_of_res', 'used_app_before', 'result', 'relation'
]

st.set_page_config(page_title="Autism Predictor", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
/* Page background */
body {
    background-color: #F9F9F9;
}

/* Card style */
.card {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 5px 5px 15px #AAAAAA;
}

/* Card headers */
.card-header {
    font-size: 20px;
    font-weight: bold;
    color: #4B0082;
    margin-bottom: 10px;
}

/* Prediction cards */
.pred-positive {
    background-color: #FFCDD2;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: #B71C1C;
    font-size: 24px;
    font-weight: bold;
}
.pred-negative {
    background-color: #C8E6C9;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: #1B5E20;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Input Sections ------------------
with st.container():
    st.markdown('<div class="card"><div class="card-header">Personal & Demographic Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", encoders['gender'].classes_)
        jaundice = st.selectbox("Jaundice", encoders['jaundice'].classes_)
        age = st.number_input("Age", min_value=0, max_value=120, value=5)
    with col2:
        ethnicity = st.selectbox("Ethnicity", encoders['ethnicity'].classes_)
        austim = st.selectbox("Austim", encoders['austim'].classes_)
        result = st.number_input("Result", min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
    with col3:
        country = st.selectbox("Country of Residence", encoders['contry_of_res'].classes_)
        used_app_before = st.selectbox("Used App Before", encoders['used_app_before'].classes_)
        relation = st.selectbox("Relation", encoders['relation'].classes_)
    st.markdown('</div>', unsafe_allow_html=True)

# A1-A10 section
with st.container():
    st.markdown('<div class="card"><div class="card-header">Answer A1–A10 (0 or 1)</div>', unsafe_allow_html=True)
    A_scores = {}
    cols = st.columns(5)
    for i in range(1, 11):
        col_idx = (i-1) % 5
        A_scores[f"A{i}_Score"] = cols[col_idx].radio(f"A{i}", options=[0,1], index=0, horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prepare Input Data ------------------
input_data = {
    'gender': gender,
    'ethnicity': ethnicity,
    'jaundice': jaundice,
    'austim': austim,
    'contry_of_res': country,
    'used_app_before': used_app_before,
    'relation': relation,
    'age': age,
    'result': result
}
input_data.update(A_scores)
input_df = pd.DataFrame([input_data])

for col in categorical_columns:
    input_df[col] = encoders[col].transform(input_df[col])

input_df = input_df[feature_order]

# ------------------ Prediction ------------------
if st.button("Predict Autism"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.markdown('<div class="pred-positive">Autism Positive</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pred-negative">Autism Negative</div>', unsafe_allow_html=True)
