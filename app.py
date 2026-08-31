from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


# Load trained SVM model
model = pickle.load(open("svc_m.pkl", "rb"))

# Load StandardScaler
scaler = pickle.load(open("std.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    alcohol = float(request.form["alcohol"])
    malic_acid = float(request.form["malic_acid"])
    ash = float(request.form["ash"])
    alcalinity_of_ash = float(request.form["alcalinity_of_ash"])
    magnesium = float(request.form["magnesium"])
    total_phenols = float(request.form["total_phenols"])
    flavanoids = float(request.form["flavanoids"])
    nonflavanoid_phenols = float(request.form["nonflavanoid_phenols"])
    proanthocyanins = float(request.form["proanthocyanins"])
    color_intensity = float(request.form["color_intensity"])
    hue = float(request.form["hue"])
    od280_od315 = float(request.form["od280_od315"])
    proline = float(request.form["proline"])


    # Create input array
    input_data = np.array([[
        alcohol,
        malic_acid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        od280_od315,
        proline
    ]])


    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_scaled)


    # Get predicted class
    predicted_class = int(prediction[0])


    return render_template(
        "index.html",
        prediction=predicted_class
    )


if __name__ == "__main__":
    app.run(debug=True)