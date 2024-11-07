import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    """Evaluates the model using accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix: \n{cm}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)

# Plot confusion matrix
def plot_confusion_matrix(cm, labels=['Real', 'Fake']):
    """Plots the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Function to preprocess and clean text
def preprocess_text(text):
    """Removes punctuation, converts to lowercase, and removes stopwords."""
    import re
    from nltk.corpus import stopwords
    import nltk

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Remove punctuation and lowercase the text
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Function to load data
def load_data(file_path):
    """Loads the dataset and preprocesses the text column."""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset from {file_path} with {df.shape[0]} rows.")
    
    # Preprocess the text column
    df['text'] = df['text'].apply(preprocess_text)
    
    return df

# Function to save model
def save_model(model, model_path):
    """Saves the trained model to a file."""
    import joblib
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
