import pandas as pd
import numpy as np
from collections import defaultdict
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_labels = [1 if i >= 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_labels)

def text_to_tfidf_vector(text, word2idx, idf):
    tf_vector = np.zeros(len(word2idx))
    words = text.lower().split()
    for word in words:
        if word in word2idx:
            idx = word2idx[word]
            tf_vector[idx] += 1
    tf_idf_vector = tf_vector * idf
    return tf_idf_vector

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, num_iter=1000, reg_param=0):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
        
    for i in range(num_iter):
        # Compute linear model
        linear_model = np.dot(X, weights) + bias
        # Apply sigmoid function
        y_predicted = sigmoid(linear_model)
            
        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (reg_param / n_samples) * weights
        db = (1 / n_samples) * np.sum(y_predicted - y)
            
        # Update weights and bias
        weights -= lr * dw
        bias -= lr * db
            
    return weights, bias

def main():
    # Load data
    data = pd.read_csv('emails.csv', header=0, names=['text', 'label'])

    # Ensure labels are integers
    data['label'] = data['label'].astype(int)

    # Convert data to lists
    texts = data['text'].tolist()
    labels = data['label'].tolist()

    # Shuffle data
    combined = list(zip(texts, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts[:], labels[:] = zip(*combined)

    # Split data into train, validation, and test sets (70%, 15%, 15%)
    total_samples = len(texts)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)

    X_train_text = texts[:train_end]
    y_train = labels[:train_end]

    X_val_text = texts[train_end:val_end]
    y_val = labels[train_end:val_end]

    X_test_text = texts[val_end:]
    y_test = labels[val_end:]

    # Build vocabulary from training data
    vocab = defaultdict(int)
    for text in X_train_text:
        words = text.lower().split()
        for word in words:
            vocab[word] += 1

    # Assign index to each word
    word2idx = {word: idx for idx, word in enumerate(vocab.keys())}
    vocab_size = len(word2idx)

    # Compute document frequencies
    doc_freq = np.zeros(vocab_size)
    for text in X_train_text:
        words = set(text.lower().split())
        for word in words:
            idx = word2idx[word]
            doc_freq[idx] += 1

    # Compute Inverse Document Frequency (IDF)
    num_docs = len(X_train_text)
    idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1  # Added 1 to prevent division by zero
    
    # Vectorize data
    X_train = np.array([text_to_tfidf_vector(text, word2idx, idf) for text in X_train_text])
    X_val = np.array([text_to_tfidf_vector(text, word2idx, idf) for text in X_val_text])
    X_test = np.array([text_to_tfidf_vector(text, word2idx, idf) for text in X_test_text])

    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Define hyperparameter grids for logistic regression
    learning_rates = [0.1, 0.01, 0.001]
    regularization_params = [0, 0.01, 0.1, 1, 10]

    best_val_accuracy = 0
    best_weights = None
    best_bias = None
    best_lr = None
    best_reg_param = None

    # Train and validate logistic regression
    for lr in learning_rates:
        for reg_param in regularization_params:
            print(f'Training with lr={lr}, reg_param={reg_param}')
            # Train model
            weights, bias = train_logistic_regression(
                X_train, y_train, lr=lr, num_iter=1000, reg_param=reg_param
            )
            # Validate model
            y_val_pred = predict(X_val, weights, bias)
            val_accuracy = np.mean(y_val_pred == y_val)
            print(f'Validation Accuracy: {val_accuracy}')
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = weights
                best_bias = bias
                best_lr = lr
                best_reg_param = reg_param

    print(f'\nBest Hyperparameters for Logistic Regression: lr={best_lr}, reg_param={best_reg_param}')
    print(f'Best Validation Accuracy: {best_val_accuracy}')

    # Evaluate the best logistic regression model on the test set
    y_test_pred = predict(X_test, best_weights, best_bias)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f'Logistic Regression Test Accuracy: {test_accuracy}')

    # Trivial baseline: Predict all emails as 'not spam' (label 0)
    baseline_pred = np.zeros_like(y_test)
    baseline_accuracy = np.mean(baseline_pred == y_test)
    print(f'Trivial Baseline Test Accuracy: {baseline_accuracy}')

    # Random guess baseline: Predict randomly among k classes
    num_classes = len(set(y_test))
    random_guess_accuracy = 1 / num_classes
    print(f'Random Guess Accuracy (1/k): {random_guess_accuracy}')

    # Save the logistic regression model parameters
    np.save('logistic_regression_weights.npy', best_weights)
    np.save('logistic_regression_bias.npy', np.array([best_bias]))
    with open('word2idx.json', 'w') as f:
        json.dump(word2idx, f)
    np.save('idf.npy', idf)

    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_val_pred_nb = nb_model.predict(X_val)
    y_test_pred_nb = nb_model.predict(X_test)
    nb_val_accuracy = np.mean(y_val_pred_nb == y_val)
    nb_test_accuracy = np.mean(y_test_pred_nb == y_test)
    print("\nMultinomial Naive Bayes:")
    print(f"Validation Accuracy: {nb_val_accuracy}")
    print(f"Test Accuracy: {nb_test_accuracy}")

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)
    y_test_pred_rf = rf_model.predict(X_test)
    rf_val_accuracy = np.mean(y_val_pred_rf == y_val)
    rf_test_accuracy = np.mean(y_test_pred_rf == y_test)
    print("\nRandom Forest Classifier:")
    print(f"Validation Accuracy: {rf_val_accuracy}")
    print(f"Test Accuracy: {rf_test_accuracy}")

    # Print a summary comparison
    print("\n=== Model Comparison ===")
    print(f"Logistic Regression (Best) - Val: {best_val_accuracy}, Test: {test_accuracy}")
    print(f"Naive Bayes               - Val: {nb_val_accuracy}, Test: {nb_test_accuracy}")
    print(f"Random Forest             - Val: {rf_val_accuracy}, Test: {rf_test_accuracy}")

if __name__ == "__main__":
    main()
