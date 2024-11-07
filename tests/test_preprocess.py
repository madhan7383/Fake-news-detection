
import unittest
from scripts.preprocess import preprocess_text

class TestPreprocessText(unittest.TestCase):

    def test_remove_punctuation(self):
        text = "Hello, World!"
        result = preprocess_text(text)
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_lowercase(self):
        text = "This is a TEST."
        result = preprocess_text(text)
        self.assertEqual(result, "test")

    def test_remove_stopwords(self):
        text = "This is a test case for preprocessing."
        result = preprocess_text(text)
        self.assertNotIn("is", result)
        self.assertNotIn("a", result)
        self.assertIn("test", result)

    def test_combined(self):
        text = "NLTK makes PREPROCESSING, easier!"
        result = preprocess_text(text)
        self.assertEqual(result, "nltk preprocessing easier")

if __name__ == '__main__':
    unittest.main()
