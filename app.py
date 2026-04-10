from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page (UI)
@app.route("/")
def home():
    return render_template("index.html")

# API (for Postman)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    transformed = vectorizer.transform([data])
    prediction = model.predict(transformed)[0]

    return jsonify({"spam": int(prediction)})

# Form submission (UI)
@app.route("/predict_form", methods=["POST"])
def predict_form():
    message = request.form["message"]
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)