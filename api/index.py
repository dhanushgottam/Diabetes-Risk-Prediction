from flask import Flask, render_template, request
import numpy as np
import joblib
import os


app = Flask(__name__, template_folder="../templates")


BASE_DIR = os.path.dirname(os.path.dirname(__file__))


model = joblib.load(os.path.join(BASE_DIR, "xgboost_diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_means = joblib.load(os.path.join(BASE_DIR, "feature_means.pkl"))


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

    # Input validation
    for feature in major_features:
        value = request.form.get(feature)

        if value is None or value == "":
            return render_template(
                "index.html",
                error="Please fill all fields before predicting."
            )

        try:
            user_values.append(float(value))
        except ValueError:
            return render_template(
                "index.html",
                error="Invalid input detected. Please check your values."
            )

    # Build full feature vector
    full_input = []
    for col in feature_means.index:
        if col in major_features:
            full_input.append(user_values[major_features.index(col)])
        else:
            full_input.append(feature_means[col])

    final_input = np.array(full_input).reshape(1, -1)

    # Scale input
    final_input_scaled = scaler.transform(final_input)

    # Predict
    prediction = model.predict(final_input_scaled)[0]
    probability = model.predict_proba(final_input_scaled)[0][1]

    result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"

    return render_template(
        "result.html",
        result=result,
        prob=round(probability * 100, 2)
    )

# REQUIRED for Vercel
handler = app