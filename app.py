from flask import Flask, request, jsonify, render_template
import pickle
import logging

app = Flask(__name__)

logging.basicConfig(filename="app.log", level=logging.INFO)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Store last 5 predictions
history = []

@app.route("/")
def home():
    return render_template("index.html", history=history)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    global history

    message = request.form["message"]

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0][1]

    result = "Spam" if prediction == 1 else "Not Spam"

    # Save history (max 5)
    history.insert(0, {
        "message": message,
        "result": result,
        "prob": round(probability * 100, 2)
    })
    history = history[:5]

    logging.info(f"{message} | {result} | {probability:.2f}")

    return render_template(
        "index.html",
        prediction=result,
        prob=round(probability * 100, 2),
        message=message,
        history=history
    )

if __name__ == "__main__":
    app.run(debug=True)