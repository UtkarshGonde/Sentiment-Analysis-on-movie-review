from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
CORS(app)  # Allow cross-origin (safe since we use only Flask)

# Load the trained sentiment analysis model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer for text preprocessing
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Sentiment prediction function
def predict_sentiment(review):
    review_seq = pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=100)
    prediction = model.predict(review_seq)
    sentiment = prediction.argmax()  # Get the index of the highest probability
    return int(sentiment)  # Return only sentiment index (no confidence value)

# Serve the main review page
@app.route("/")
def home():
    return render_template("index.html")

# Serve the main review page
@app.route("/review")
def review():
    return render_template("review.html")


# API for sentiment prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    
    if not review:
        return jsonify({"error": "No review provided"}), 400

    sentiment = predict_sentiment(review)

    return jsonify({"sentiment": sentiment})  # Only return the sentiment index

if __name__ == "__main__":
    app.run(debug=True)
