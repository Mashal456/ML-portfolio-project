pip install joblib

import joblib
joblib.dump(logreg, "ML-end-to-end")

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f'model saved to {filename}')

save_model(logreg, "ML-end-to-end.pkl")

def load_model(filename):
    model = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model

load_model("ML-end-to-end.pkl")
