import numpy as np

# Training Data

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

# Initialize w and b

w = 0.0
b = 0.0

learning_rate = 0.1
epochs = 1000
n = len(X)

# Sigmoid Function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Training using Gradient Descent

for i in range(epochs):

    # Calculate z
    z = w * X + b

    # Apply Sigmoid
    prediction = sigmoid(z)

    # Calculate Error
    error = prediction - y

    # Calculate Cost
    cost = -(1 / n) * np.sum(
        y * np.log(prediction + 1e-9) +
        (1 - y) * np.log(1 - prediction + 1e-9)
    )

    # Calculate Gradients
    dw = (1 / n) * np.sum(error * X)
    db = (1 / n) * np.sum(error)

    # Update w and b
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print cost occasionally
    if i % 100 == 0:
        print("Epoch:", i, "Cost:", cost)


# Trained Model

print("\nTrained Model")
print("Weight (w):", w)
print("Bias (b):", b)



# Prediction Function

def predict(x):
    z = w * x + b

    # Probability
    probability = sigmoid(z)

    # Apply threshold
    if probability >= 0.5:
        result = 1
    else:
        result = 0

    return probability, result

# New Input
new_input = 4.5

probability, result = predict(new_input)

print("\nNew Input:", new_input)
print("Probability:", probability)
print("Predicted Class:", result)