from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("Titanic Prediction.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Define mappings for Age & Fare bins
    age_mapping = {
        "0-17": 0.0,
        "18-32": 1.0,
        "33-48": 2.0,
        "49-64": 3.0,
        "65-80": 4.0,
    }
    fare_mapping = {
        "0-7": 0.0,
        "8-14": 1.0,
        "15-42": 2.0,
        "43-": 3.0,
    }

    try:
        # Collect features from form
        features = [
            float(request.form["Pclass"]),
            float(request.form["Sex"]),
            float(request.form["Embarked"]),
            age_mapping[request.form["Age"]],
            fare_mapping[request.form["Fare"]],
            float(request.form["Fam_type"]),
            float(request.form["Title"]),
        ]

        # Prediction (0 = not survived, 1 = survived)
        prediction = model.predict([features])[0]

        # Probability of survival
        proba = model.predict_proba([features])[0][1] * 100

        if prediction == 1:
            result = f"✅ Survived (Chance: {proba:.2f}%)"
        else:
            result = f"❌ Did not survive (Chance: {proba:.2f}%)"

    except Exception as e:
        result = f"⚠️ Error: {str(e)}"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
