from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# โหลด model + label encoder
model, le = joblib.load("lamp_ai_full.pkl")

@app.route("/")
def home():
    return "Smart Lamp AI API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([[
        data["zone"],
        data["current"],
        data["power"],
        data["broken"]
    ]])

    pred = model.predict(X)
    status = le.inverse_transform(pred)[0]

    return jsonify({"prediction": status})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
