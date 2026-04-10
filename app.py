from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Spam Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    transformed = vectorizer.transform([data])
    prediction = model.predict(transformed)[0]

    return jsonify({
        "input": data,
        "spam": int(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)