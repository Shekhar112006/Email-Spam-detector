from flask import Flask, request, jsonify, render_template
import pickle
import logging

app = Flask(__name__)

# Logging setup
logging.basicConfig(filename="app.log", level=logging.INFO)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Store last 5 predictions
history = []

def get_counts():
    spam_count = sum(1 for item in history if item["result"] == "Spam")
    ham_count = sum(1 for item in history if item["result"] == "Not Spam")
    return spam_count, ham_count

@app.route("/")
def home():
    spam_count, ham_count = get_counts()
    return render_template(
        "index.html",
        history=history,
        spam_count=spam_count,
        ham_count=ham_count
    )

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

    # Logging
    logging.info(f"{message} | {result} | {probability:.2f}")

    spam_count, ham_count = get_counts()

    return render_template(
        "index.html",
        prediction=result,
        prob=round(probability * 100, 2),
        message=message,
        history=history,
        spam_count=spam_count,
        ham_count=ham_count
    )

if __name__ == "__main__":
    app.run(debug=True)