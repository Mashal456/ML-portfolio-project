!pip install gradio

import gradio as gr

loaded_model = load_model("ML-end-to-end.pkl")
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


demo = gr.Interface(
    fn=predict_heart_attack,
    inputs=[
        gr.Textbox(label="Age (in years)", placeholder="Enter age"),
        gr.Textbox(label="Sex (1 = male, 0 = female)", placeholder="Enter sex (1 or 0)"),
        gr.Textbox(label="Chest Pain Type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 0 = asymptomatic)", placeholder="Enter chest pain type"),
        gr.Textbox(label="Resting Blood Pressure (in mm Hg)", placeholder="Enter resting blood pressure"),
        gr.Textbox(label="Serum Cholesterol (mg/dl)", placeholder="Enter serum cholesterol"),
        gr.Textbox(label="Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", placeholder="Enter fasting blood sugar"),
        gr.Textbox(label="Resting ECG (1 = normal, 2 = ST-T wave abnormality, 0 = hypertrophy)", placeholder="Enter resting ECG"),
        gr.Textbox(label="Maximum Heart Rate Achieved", placeholder="Enter maximum heart rate"),
        gr.Textbox(label="Exercise Induced Angina (1 = yes, 0 = no)", placeholder="Enter exercise induced angina"),
        gr.Textbox(label="ST Depression Induced by Exercise", placeholder="Enter ST depression"),
        gr.Textbox(label="Slope of the Peak Exercise ST Segment (2 = upsloping, 1 = flat, 0 = downsloping)", placeholder="Enter slope of ST segment"),
        gr.Textbox(label="Number of Major Vessels Colored by Fluoroscopy (0-3)", placeholder="Enter number of major vessels"),
        gr.Textbox(label="Thalassemia (2 = normal, 1 = fixed defect, 3 = reversable defect)", placeholder="Enter thalassemia")
    ],
    outputs=['text']
)

demo.launch(share=True)

Deployment on hugging face: https://huggingface.co/spaces/Mashal456/ML-project_deployment


