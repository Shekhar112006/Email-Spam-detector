from flask import Flask, request, jsonify, render_template
import pickle
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(filename="app.log", level=logging.INFO)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():
    message = request.form["message"]

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0][1]

    result = "Spam" if prediction == 1 else "Not Spam"

    # Logging user input + result
    logging.info(f"Message: {message} | Result: {result} | Prob: {probability:.2f}")

    return render_template(
        "index.html",
        prediction=result,
        prob=round(probability * 100, 2),
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)