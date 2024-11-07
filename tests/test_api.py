import unittest
import requests

class TestFakeNewsAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"

    def test_health_check(self):
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "OK")

    def test_predict_fake(self):
        test_data = {"text": "This is a fake news article."}
        response = requests.post(f"{self.BASE_URL}/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json().get("label"), ["Fake", "Real"])

    def test_predict_real(self):
        test_data = {"text": "This is a legitimate news article."}
        response = requests.post(f"{self.BASE_URL}/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json().get("label"), ["Fake", "Real"])

    def test_invalid_request(self):
        response = requests.post(f"{self.BASE_URL}/predict", json={})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
