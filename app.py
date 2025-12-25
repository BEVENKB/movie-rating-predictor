from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

with open("movie_rating_model.pkl", "rb") as f:
    model = joblib.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        budget = float(request.form["budget"])
        duration = float(request.form["duration"])
        genre = float(request.form["genre"])

        features = np.array([[budget, duration, genre]])
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
