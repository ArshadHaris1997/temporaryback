import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("toss_decision_model_v2.pkl")

# Dummy list of teams and venues for simulation
teams = ['India', 'Australia', 'Pakistan', 'England', 'South Africa']
venues = ['MCG', 'Eden Gardens', 'Wankhede', 'Lords', 'SCG', 'Old Trafford']

# Streamlit App Config
st.set_page_config(page_title="Toss Decision Recommender", page_icon="🧢")
st.title("🧢 Captaincy Toss Decision Recommender")
st.markdown("Predict the better toss decision (bat or bowl) based on venue, teams, and historical match patterns.")

# User Inputs
venue = st.selectbox("🏟️ Venue", venues)
toss_winner = st.selectbox("🏏 Toss Winner", teams)
opponent = st.selectbox("👊 Opponent", [team for team in teams if team != toss_winner])
toss_decision = st.radio("🤔 Toss Decision", ['bat', 'bowl'])

# Build input DataFrame
input_df = pd.DataFrame({
    'venue': [venue],
    'toss_winner': [toss_winner],
    'bowling_team': [opponent],
    'toss_decision': [toss_decision]
})

# Encode for model
input_encoded = pd.get_dummies(input_df)
for col in model.feature_names_in_:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model.feature_names_in_]

# Make prediction
prediction = model.predict(input_encoded)[0]
result = "✅ Likely Win" if prediction == 1 else "❌ Likely Loss"
st.subheader(f"🎯 Prediction: {result}")

# Optional Probabilities
proba = model.predict_proba(input_encoded)[0]
st.markdown(f"**Win Probability:** {proba[1]*100:.1f}%  |  **Loss Probability:** {proba[0]*100:.1f}%")
