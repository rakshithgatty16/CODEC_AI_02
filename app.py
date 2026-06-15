from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    senior = int(request.form["senior"])
    partner = int(request.form["partner"])
    dependents = int(request.form["dependents"])
    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])
    total = float(request.form["total"])
    contract = int(request.form["contract"])
    payment = int(request.form["payment"])

    features = np.array([[gender,
                          senior,
                          partner,
                          dependents,
                          tenure,
                          monthly,
                          total,
                          contract,
                          payment]])

    prediction = model.predict(features)

    result = "Customer Will Leave" if prediction[0] == 1 else "Customer Will Stay"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)