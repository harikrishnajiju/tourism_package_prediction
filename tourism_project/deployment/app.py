import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="<YOUR_HF_USERNAME>/tourism-package-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("🌿 Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict whether they will purchase the Wellness Tourism Package.")

# --- Customer Details ---
st.subheader("Customer Information")
age = st.number_input("Age", min_value=18, max_value=80, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=20000, step=500)
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_trips = st.number_input("Number of Trips Per Year", min_value=0, max_value=20, value=3)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children = st.number_input("Number of Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0)
preferred_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "AVP", "VP"])

# --- Interaction Data ---
st.subheader("Sales Interaction")
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15)

# Encode categoricals (must match prep.py LabelEncoder order)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

contact_map = {"Company Invited": 0, "Self Inquiry": 1}
occ_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map = {"Fe Male": 0, "Female": 1, "Male": 2}
product_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
desig_map = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact_map[type_of_contact],
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occ_map[occupation],
    "Gender": gender_map[gender],
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_map[product_pitched],
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_map[marital_status],
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": desig_map[designation],
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.success(f"✅ Customer is likely to purchase the package! (Probability: {probability:.1%})")
    else:
        st.warning(f"❌ Customer is unlikely to purchase. (Probability: {probability:.1%})")
