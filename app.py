from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

crop_model = pickle.load(open(MODELS_DIR / "crop_recommendation_rf.pkl", "rb"))
crop_scaler = pickle.load(open(MODELS_DIR / "crop_recommendation_scaler.pkl", "rb"))
crop_encoder = pickle.load(open(MODELS_DIR / "crop_recommendation_encoder.pkl", "rb"))

fertilizer_model = pickle.load(open(MODELS_DIR / "fertilizer_recommendation_model.pkl", "rb"))
fertilizer_encoder = pickle.load(open(MODELS_DIR / "fertilizer_recommendation_encoder.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crop")
def crop_page():
    return render_template("crop.html")


@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    input_data_scaled = crop_scaler.transform(input_data)
    prediction = crop_model.predict(input_data_scaled)
    predicted_crop = crop_encoder.inverse_transform(prediction)[0]

    return render_template(
        "crop.html",
        predicted_crop=predicted_crop,
        title="Crop Recommendation",
        N=N,
        P=P,
        K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall,
    )


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    temperature = float(request.form["temperature"])
    moisture = float(request.form["moisture"])
    rainfall = float(request.form["rainfall"])
    ph = float(request.form["ph"])
    nitrogen = float(request.form["nitrogen"])
    phosphorous = float(request.form["phosphorous"])
    potassium = float(request.form["potassium"])
    carbon = float(request.form["carbon"])
    soil = request.form["soil"]
    crop = request.form["crop"]
    remark = request.form.get("remark", "General recommendation")

    input_df = pd.DataFrame(
        [{
            "Temperature": temperature,
            "Moisture": moisture,
            "Rainfall": rainfall,
            "PH": ph,
            "Nitrogen": nitrogen,
            "Phosphorous": phosphorous,
            "Potassium": potassium,
            "Carbon": carbon,
            "Soil": soil,
            "Crop": crop,
            "Remark": remark,
        }]
    )

    prediction = fertilizer_model.predict(input_df)
    predicted_fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

    return render_template(
        "fertilizer.html",
        predicted_fertilizer=predicted_fertilizer,
        title="Fertilizer Recommendation",
        temperature=temperature,
        moisture=moisture,
        rainfall=rainfall,
        ph=ph,
        nitrogen=nitrogen,
        phosphorous=phosphorous,
        potassium=potassium,
        carbon=carbon,
        soil=soil,
        crop=crop,
        remark=remark,
    )


if __name__ == "__main__":
    app.run(debug=True)