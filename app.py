from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

history = []

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    prob = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]

        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data).max()

        result = "Spam" if prediction == 1 else "Not Spam"

        history.insert(0, {"text": message, "result": result})
        if len(history) > 5:
            history.pop()

        prob = round(probability * 100, 2)

    spam_count = sum(1 for item in history if item["result"] == "Spam")
    ham_count = sum(1 for item in history if item["result"] == "Not Spam")

    return render_template(
        "index.html",
        prediction=result,
        prob=prob,
        message=message,
        history=history,
        spam_count=spam_count,
        ham_count=ham_count
    )

if __name__ == "__main__":
    app.run(debug=True)