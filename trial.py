import streamlit as st
import pandas as pd
import joblib

# Define the predict_unicorn function
def predict_unicorn(model, total_funding_c, EBIT_c, last_round_size_c, revenue_c, revenue_growh, employee_growth_6, employee_growth_12, num_funding_rounds, num_shareholders):
    """
    Predicts if a startup is a unicorn given its features using a trained model.

    Args:
    model: Trained classification model.
    total_funding: Total funding till date.
    last_round_size: Amount raised during last funding round.
    revenue: Revenue for latest financial year.
    revenue_growth: Revenue growth compared to last financial year.
    EBIT: Earnings before interest and tax.
    employee_growth_6: Employee growth past 6 months.
    employee_growth_12: Employee growth past 12 months.
    num_funding_rounds: Number of funding rounds.
    num_shareholders: Number of shareholders.

    Returns:
    A string indicating whether the startup is a unicorn or not.
    """
    # Create a dataframe from the input data
    input_data = pd.DataFrame({
        'total_funding_c': [total_funding_c],
        'EBIT_c': [EBIT_c],
        'last_round_size_c': [last_round_size_c],
        'revenue_c': [revenue_c],
        'revenue_growh': [revenue_growh],
        'employee_growth_6': [employee_growth_6],
        'employee_growth_12': [employee_growth_12],
        'num_funding_rounds': [num_funding_rounds],
        'num_shareholders': [num_shareholders]
    })

    # Make prediction using the trained model
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "Unicorn"
    else:
        return "Not a unicorn"

# Load the trained model
rf = joblib.load("rf.joblib")

# Create a streamlit web app
st.title("Unicorn Startup Predictor")
st.markdown("Enter the details of the startup below to predict if it will become a unicorn.")

# Create input fields for startup features
total_funding_c = st.number_input("Total funding till date")
EBIT_c = st.number_input("EBIT")
last_round_size_c = st.number_input("Amount raised during last funding round")
revenue_c = st.number_input("Revenue for latest financial year")
revenue_growh = st.number_input("Revenue growth compared to last financial year")
employee_growth_6 = st.number_input("Employee growth past 6 months")
employee_growth_12 = st.number_input("Employee growth past 12 months")
num_funding_rounds = st.number_input("Number of funding rounds")
num_shareholders = st.number_input("Number of shareholders")

# Create a button to submit the inputs and make prediction
if st.button("Predict"):
    # Create a dictionary of startup features
    startup = {
        'model': rf,
        'total_funding_c': total_funding_c, 
        'EBIT_c': EBIT_c, 
        'last_round_size_c': last_round_size_c, 
        'revenue_c': revenue_c, 
        'revenue_growh': revenue_growh,
        'employee_growth_6': employee_growth_6, 
        'employee_growth_12': employee_growth_12, 
        'num_funding_rounds': num_funding_rounds, 
        'num_shareholders': num_shareholders
    }

    # Call the predict_unicorn function with the startup dictionary
    is_unicorn = predict_unicorn(**startup)

    # Display the result on the streamlit app
    if is_unicorn == "Unicorn":
        st.write("Congratulations, the startup is going to be a Unicorn!")
    else:
        st.write("Sorry, the startup is not a Unicorn.")