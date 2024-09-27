pip install fastapi nest-asyncio pyngrok uvicorn

!ngrok config add-authtoken 2mG9izhmfMLeYdETgjukYbcI58L_4ujwD8F8id88Qz55ZEvuR


from fastapi import FastAPI, HTTPException
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = load_model("ML-end-to-end.pkl")

# Create FastAPI instance
app = FastAPI()

# Define a function to make predictions
def predict_heart_attack(age, sex, chest_pain_type, resting_blood_pressure, serum_cholesterol, fasting_blood_sugar,
                         rest_ecg, max_heart_rate, exercise_induced_angina, st_depression, slope_of_st, num_major_vessels, thal):
    # Prepare input data
    input_data = {
        'age': [age], 'sex': [sex], 'cp': [chest_pain_type], 'trtbps': [resting_blood_pressure],
        'chol': [serum_cholesterol], 'fbs': [fasting_blood_sugar], 'restecg': [rest_ecg],
        'thalachh': [max_heart_rate], 'exng': [exercise_induced_angina], 'oldpeak': [st_depression],
        'slp': [slope_of_st], 'caa': [num_major_vessels], 'thall': [thal]
    }

    input_df = pd.DataFrame(input_data)

    # Encoding: Ensure columns match those used during training
    input_df_encoded = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slp', 'caa', 'thall'], drop_first=True)

    # Ensure the input_df_encoded has the same columns as the model expects
    missing_cols = set(model.feature_names_in_) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[model.feature_names_in_]

    # Make prediction
    try:
        prediction = model.predict(input_df_encoded)
        prediction_proba = model.predict_proba(input_df_encoded)
        return {
            'prediction': 'The person is likely to have a heart attack' if prediction[0] == 1 else 'The person is unlikely to have a heart attack',
            'probability': float(prediction_proba[0][1])  # Probability of the positive class
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define FastAPI GET endpoint for prediction
@app.get('/predict')
async def predict(age: int, sex: int, chest_pain_type: int, resting_blood_pressure: int, serum_cholesterol: int,
                  fasting_blood_sugar: int, rest_ecg: int, max_heart_rate: int, exercise_induced_angina: int,
                  st_depression: float, slope_of_st: int, num_major_vessels: int, thal: int):
    # Call prediction function
    result = predict_heart_attack(age, sex, chest_pain_type, resting_blood_pressure, serum_cholesterol, fasting_blood_sugar,
                                  rest_ecg, max_heart_rate, exercise_induced_angina, st_depression, slope_of_st, num_major_vessels, thal)
    return result

# Example route to test if the API is working
@app.get('/')
async def index():
    return {'message': 'Welcome to the Heart Attack Prediction API'}

# Setup ngrok for public access
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

# Required for running in notebooks or certain environments
nest_asyncio.apply()

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, port=8000)
