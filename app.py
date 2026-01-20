@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"status": "AI server online"})

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

        return jsonify({"prediction": str(status)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
