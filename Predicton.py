import streamlit as st
import pandas as pd
# import numpy as np
import pickle
from datetime import date

# 1. Load the trained Decision Tree Model
try:
    with open('DTR_model.pkl', 'rb') as file:
        model_DT = pickle.load(file)
except FileNotFoundError:
    st.error("Error: The trained model file 'DT_model.pkl' was not found.")
    st.stop()

# 2. Define Feature Mappings (CRUCIAL STEP)
# Based on your training script, 'storey_range' was ordinal encoded.
# The following is a manual mapping based on the unique values and the assumed encoding order.
# In a real-world scenario, you MUST save the fitted encoder object.
STOREY_RANGES = [
    '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
    '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
    '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
]
# Create a dictionary to map the string to the ordinal encoded number (0 to 16)
storey_range_mapping = {range_val: i for i, range_val in enumerate(STOREY_RANGES)}

# Define all possible categories for the dropped/encoded columns (for demonstration purposes)
# These are the original categorical columns that were encoded and then dropped,
# but an app might want to collect them anyway.
TOWNS = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
         'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG',
         'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']

FLAT_TYPES = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']


# 3. Streamlit Interface Setup
st.set_page_config(page_title="🇸🇬 Singapore HDB Resale Price Predictor", layout="centered")

st.title("🇸🇬 HDB Resale Price Prediction")
st.markdown("Use this tool to predict the estimated resale price of a Singapore HDB flat based on its attributes.")

st.sidebar.header("Flat Details Input")

# --- Input Fields ---

# Floor Area
floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", min_value=30.0, max_value=150.0, value=90.0, step=1.0)

# Storey Range (The only encoded categorical feature used in the final DT model features)
storey_range_input = st.sidebar.selectbox("Storey Range", options=STOREY_RANGES)

# Lease Commence Date
current_year = date.today().year
lease_commence_date = st.sidebar.slider(
    "Lease Commencement Year",
    min_value=1960,
    max_value=current_year,
    value=1985
)

# Year and Month (Derived from the current date for the prediction context)
# The model needs 'year' and 'month' as integers. We will use the current date
# or a date selected by the user to represent the prediction point in time.
st.sidebar.subheader("Prediction Month/Year")
year_input = st.sidebar.number_input("Year of Sale", min_value=1990, max_value=current_year + 5, value=current_year)
month_input = st.sidebar.number_input("Month of Sale (1-12)", min_value=1, max_value=12, value=11)


# --- For Demonstration/Completeness (Though not used by the saved DT model) ---
with st.expander("Show Features Excluded from Current Model (DT_model.pkl)"):
    st.selectbox("Town (Excluded)", options=TOWNS)
    st.selectbox("Flat Type (Excluded)", options=FLAT_TYPES)
    st.info("These features were dropped before the Decision Tree model was trained, but an optimal model would use them.")


# 4. Prediction Logic
if st.button("Predict Resale Price"):
    # --- Pre-processing User Input ---

    # Get the ordinal encoded value for storey_range
    # The encoding value must be what the model expects (0 to 16 in this case)
    storey_range_encoded = storey_range_mapping.get(storey_range_input)

    # Create a DataFrame for the prediction
    # The order of columns MUST match the training feature order:
    # ['month', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'year']
    data = {
        'month': [month_input],
        'storey_range': [storey_range_encoded], # Encoded value
        'floor_area_sqm': [floor_area_sqm],
        'lease_commence_date': [lease_commence_date],
        'year': [year_input]
    }

    input_df = pd.DataFrame(data)

    # Display the processed input for verification
    st.subheader("Input Data for Prediction")
    st.dataframe(input_df)

    # --- Make Prediction ---
    try:
        predicted_price = model_DT.predict(input_df)[0]

        # 5. Display Result
        st.success("---")
        st.header("🔮 Predicted Resale Price")
        st.balloons()
        st.markdown(f"The estimated resale price is:")
        st.metric(label="Predicted Price", value=f"S$ {predicted_price:,.2f}")
        st.caption("Note: This prediction is based on the Decision Tree Regressor model trained and saved in your script.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the 'DT_model.pkl' file is correctly trained and the feature order/types match.")