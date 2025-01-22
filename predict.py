import numpy as np
import json
from train import text_to_tfidf_vector, predict

def load_model(weights_file, bias_file, word2idx_file, idf_file):
    weights = np.load(weights_file)
    bias = np.load(bias_file)[0]
    with open(word2idx_file, 'r') as f:
        word2idx = json.load(f)
    idf = np.load(idf_file)
    return weights, bias, word2idx, idf

def predict_spam(text):
    vector = text_to_tfidf_vector(text, word2idx, idf)
    y_pred = predict(np.array([vector]), weights, bias)
    return y_pred[0]

# Load the model
weights, bias, word2idx, idf = load_model(
    'logistic_regression_weights.npy',
    'logistic_regression_bias.npy',
    'word2idx.json',
    'idf.npy'
)

# Example usage
email_text = "Your Uber code is 1487. Never share this code. Reply STOP ALL to unsubscribe."
result = predict_spam(email_text)
print("Spam" if result == 1 else "Not Spam")
