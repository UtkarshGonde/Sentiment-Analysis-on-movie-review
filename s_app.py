from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizerfrom flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Prediction function
def predict_sentiment(review):
    review_seq = pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=100)
    prediction = model.predict(review_seq)
    sentiment = prediction.argmax()
    return sentiment

# Serve HTML page
@app.route("/")
def home():
    return render_template("index1.html")  # Make sure this is in the 'templates' folder

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    if not review:
        return jsonify({"error": "No review provided"}), 400

    sentiment = predict_sentiment(review)
    return jsonify({"sentiment": int(sentiment)})

if __name__ == "__main__":
    app.run(debug=True)

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Prediction function
def predict_sentiment(review):
    review_seq = pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=100)
    prediction = model.predict(review_seq)
    sentiment = prediction.argmax()
    return sentiment

# Serve HTML
@app.route("/")
def home():
    return render_template("index1.html")

# API Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data.get("review", "")
    if not review:
        return jsonify({"error": "No review provided"}), 400

    sentiment = predict_sentiment(review)
    return jsonify({"sentiment": int(sentiment)})

if __name__ == "__main__":
    app.run(debug=True)
