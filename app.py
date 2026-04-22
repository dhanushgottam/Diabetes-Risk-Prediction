from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model files (normal paths)
model = joblib.load("xgboost_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_means = joblib.load("feature_means.pkl")

major_features = [
    "BMI",
    "HighBP",
    "HighChol",
    "HvyAlcoholConsump",
    "GenHlth",
    "Age",
    "HeartDiseaseorAttack",
    "DiffWalk",
    "PhysActivity",
    "Smoker"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_values = []

    for feature in major_features:
        value = request.form.get(feature)

        if value is None or value == "":
            return render_template("index.html", error="Fill all fields")

        user_values.append(float(value))

    full_input = []

    for col in feature_means.index:
        if col in major_features:
            full_input.append(user_values[major_features.index(col)])
        else:
            full_input.append(feature_means[col])

    final_input = np.array(full_input).reshape(1, -1)
    final_input_scaled = scaler.transform(final_input)

    prediction = model.predict(final_input_scaled)[0]
    probability = model.predict_proba(final_input_scaled)[0][1]

    result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"

    return render_template("result.html", result=result, prob=round(probability*100,2))