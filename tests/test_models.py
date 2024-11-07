
import unittest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class TestModels(unittest.TestCase):

    def setUp(self):
        # Load test data and the trained model
        self.model = joblib.load('models/best_fake_news_model.pkl')
        self.test_data = pd.read_csv('data/Fake.csv').sample(10)  # Sample for quick test

        # Preprocess the test data as required by the model
        self.test_data['text'] = self.test_data['text'].apply(self.model.named_steps['tfidf'].build_analyzer())
        self.X_test = self.test_data['text']
        self.y_test = self.test_data['label']

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, Pipeline, "Model is not a Pipeline instance.")

    def test_predictions(self):
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        self.assertGreater(acc, 0.80, "Model accuracy should be > 80%.")

    def test_saved_model_integrity(self):
        self.assertTrue(self.model, "Model did not load properly from disk.")
        
if __name__ == '__main__':
    unittest.main()
