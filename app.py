from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/information_diffusion_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        total_interactions = float(request.form["total_interactions"])
        retweets = float(request.form["retweets"])
        mentions = float(request.form["mentions"])
        replies = float(request.form["replies"])

        # Input must be in the same order used during training
        input_data = [[
            total_interactions,
            retweets,
            mentions,
            replies
        ]]

        prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)