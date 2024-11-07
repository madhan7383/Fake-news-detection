import joblib
from scripts.preprocess import preprocess_text

def predict_fake_news(text):
    # Load the model
    model = joblib.load('models/best_fake_news_model.pkl')
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Predict
    prediction = model.predict([processed_text])
    label = "Fake" if prediction[0] == 1 else "Real"
    return label

if __name__ == "__main__":
    sample_text = "Sample news article text for prediction."
    print(f"Prediction: {predict_fake_news(sample_text)}")
