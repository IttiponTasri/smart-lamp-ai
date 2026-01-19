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

# health check สำหรับ browser / render
@app.route("/predict", methods=["GET"])
def predict_check():
    return jsonify({"status": "AI server online"})

# prediction จริง
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        X = np.array([[ 
            float(data.get("zone", 0)),
            float(data.get("current", 0)),
            float(data.get("power", 0)),
            float(data.get("broken", 0))
        ]])

        pred = model.predict(X)
        status = le.inverse_transform(pred)[0]

        return jsonify({
            "prediction": str(status)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
