from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model, scaler, and feature means
model = joblib.load("xgboost_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_means = joblib.load("feature_means.pkl")

app = Flask(__name__)

# 10 major inputs
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

    # 🔴 SAFE INPUT HANDLING
    for feature in major_features:
        value = request.form.get(feature)

        # 🚨 If any field is empty → stop submission
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

    # Step 2 → Create full 21-feature input
    full_input = []

    for col in feature_means.index:
        if col in major_features:
            full_input.append(user_values[major_features.index(col)])
        else:
            full_input.append(feature_means[col])

    final_input = np.array(full_input).reshape(1, -1)

    # Step 3 → Scale
    final_input_scaled = scaler.transform(final_input)

    # Step 4 → Predict
    prediction = model.predict(final_input_scaled)[0]
    probability = model.predict_proba(final_input_scaled)[0][1]

    result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"

    return render_template(
        "result.html",
        result=result,
        prob=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)