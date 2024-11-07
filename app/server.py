from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('models/best_fake_news_model.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid input"}), 400

    text = [request.json['text']]  # Wrap input in a list for the model
    prediction = model.predict(text)[0]
    label = "Fake" if prediction == 1 else "Real"

    return jsonify({"label": label}), 200

if __name__ == '__main__':
    app.run(debug=True)
