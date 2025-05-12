import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('../model/rf_model.pkl')
    scaler = joblib.load('../model/scaler.pkl')
    return model, scaler

# Function to preprocess input data
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, title, scaler):
    # Create a dictionary with the input data
    data = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'FamilySize': sibsp + parch + 1
    }
    
    # Add one-hot encoded features for Sex
    data['Sex_female'] = 1 if sex == 'female' else 0
    data['Sex_male'] = 1 if sex == 'male' else 0
    
    # Add one-hot encoded features for Embarked
    data['Embarked_C'] = 1 if embarked == 'C' else 0
    data['Embarked_Q'] = 1 if embarked == 'Q' else 0
    data['Embarked_S'] = 1 if embarked == 'S' else 0
    
    # Add one-hot encoded features for Title
    all_titles = ['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 
                 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir']
    
    for t in all_titles:
        data[f'Title_{t}'] = 1 if title == t else 0
    
    # Create fare bins
    if fare <= 7.91:
        fare_bin = 'Low'
    elif fare <= 14.454:
        fare_bin = 'Mid'
    elif fare <= 31.0:
        fare_bin = 'Mid-High'
    else:
        fare_bin = 'High'
    
    # Add one-hot encoded features for FareBin
    data['FareBin_Low'] = 1 if fare_bin == 'Low' else 0
    data['FareBin_Mid'] = 1 if fare_bin == 'Mid' else 0
    data['FareBin_Mid-High'] = 1 if fare_bin == 'Mid-High' else 0
    data['FareBin_High'] = 1 if fare_bin == 'High' else 0
    
    # Create age bins
    if age < 20:
        age_bin = 'Young'
    elif age < 40:
        age_bin = 'Adult'
    elif age < 60:
        age_bin = 'Middle'
    else:
        age_bin = 'Senior'
    
    # Add one-hot encoded features for AgeBin
    data['AgeBin_Young'] = 1 if age_bin == 'Young' else 0
    data['AgeBin_Adult'] = 1 if age_bin == 'Adult' else 0
    data['AgeBin_Middle'] = 1 if age_bin == 'Middle' else 0
    data['AgeBin_Senior'] = 1 if age_bin == 'Senior' else 0
    
    # Create DataFrame with all features in the correct order
    features = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
        'Sex_female', 'Sex_male',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ]
    
    # Add all title features
    features.extend([f'Title_{t}' for t in all_titles])
    
    # Add fare bin features
    features.extend(['FareBin_Low', 'FareBin_Mid', 'FareBin_Mid-High', 'FareBin_High'])
    
    # Add age bin features
    features.extend(['AgeBin_Young', 'AgeBin_Adult', 'AgeBin_Middle', 'AgeBin_Senior'])
    
    df = pd.DataFrame([data])[features]
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    return df_scaled

# Main app
def main():
    st.title("🚢 Titanic Survival Predictor")
    st.write("""
    This app predicts whether a passenger would have survived the Titanic disaster based on their characteristics.
    Enter the passenger details below to make a prediction.
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Passenger Information")
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Number of Parents/Children", 0, 10, 0)
    
    with col2:
        st.subheader("Additional Information")
        fare = st.number_input("Fare", 0.0, 500.0, 50.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Col", "Major", "Capt", "Don", "Jonkheer", "Countess", "Lady", "Sir", "Mlle", "Mme", "Ms"])
    
    # Load model and scaler
    model, scaler = load_model()
    
    # Make prediction when button is clicked
    if st.button("Predict Survival"):
        # Preprocess input
        input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, title, scaler)
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.success("The passenger would have survived! 🎉")
        else:
            st.error("The passenger would not have survived 😢")
        
        # Display probability
        st.write(f"Survival Probability: {probability[0][1]:.2%}")
        
        # Display feature importance
        st.subheader("Top Factors Affecting Survival")
        feature_importance = pd.DataFrame({
            'Feature': ['Passenger Class', 'Age', 'Sex', 'Fare', 'Family Size'],
            'Importance': [0.2, 0.15, 0.25, 0.15, 0.25]  # Example values
        })
        st.bar_chart(feature_importance.set_index('Feature'))

if __name__ == "__main__":
    main() 