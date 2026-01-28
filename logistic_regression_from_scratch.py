import numpy as np

# -----------------------------
# Sigmoid Function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------
# Cost Function (Log Loss)
# -----------------------------
def compute_cost(X, y, weights, bias):
    m = len(y)
    z = np.dot(X, weights) + bias
    y_hat = sigmoid(z)
    
    cost = -(1/m) * np.sum(
        y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    )
    return cost


# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(X, y, weights, bias, lr, iterations):
    m = len(y)
    
    for _ in range(iterations):
        z = np.dot(X, weights) + bias
        y_hat = sigmoid(z)
        
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        
        weights -= lr * dw
        bias -= lr * db
        
    return weights, bias


# -----------------------------
# Training Function
# -----------------------------
def train_logistic_regression(X, y, lr=0.01, iterations=1000):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    weights, bias = gradient_descent(X, y, weights, bias, lr, iterations)
    return weights, bias


# -----------------------------
# Prediction
# -----------------------------
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return [1 if p >= 0.5 else 0 for p in probs]


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Sample Data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1])

    # Train model
    weights, bias = train_logistic_regression(X, y)

    # Predictions
    predictions = predict(X, weights, bias)

    print("Weights:", weights)
    print("Bias:", bias)
    print("Predictions:", predictions)
