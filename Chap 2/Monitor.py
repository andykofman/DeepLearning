import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([
    [2.0, 3.5, 1.1],
    [1.0, 2.0, 3.0],
    [4.0, 1.5, 2.2],
    [3.0, 2.5, 1.8]
])
y_batch = np.array([1.0, 0.0, 1.0, 0.0])

# Initialize parameters
w = np.array([0.5, -0.3, 0.8])
b = 38
learning_rate = 0.01
num_epochs = 100  # Keep small for clean plots

# Containers for logging
losses = []
biases = []
weights = []
grad_w = []
grad_b = []
predictions_per_epoch = []

# Prediction function
def predict(X, w, b):
    return np.dot(X, w) + b

# MSE Loss
def calculate_mse(p_batch, y_batch):
    return np.mean((p_batch - y_batch) ** 2)

# Training loop
for epoch in range(num_epochs):
    p_batch = predict(X, w, b)
    loss = calculate_mse(p_batch, y_batch)

    # Gradients
    error = y_batch - p_batch
    dw = -2 * np.dot(X.T, error) / len(X)
    db = -2 * np.mean(error)

    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Logging
    losses.append(loss)
    biases.append(b)
    weights.append(w.copy())
    grad_w.append(dw.copy())
    grad_b.append(db)
    predictions_per_epoch.append(p_batch.copy())

# Plotting section
epochs = range(num_epochs)

# 1. Loss over time
plt.figure(figsize=(10, 4))
plt.plot(epochs, losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs")
plt.grid()
plt.legend()
plt.show()

# 2. Weight updates
weights = np.array(weights)
plt.figure(figsize=(10, 4))
for i in range(weights.shape[1]):
    plt.plot(epochs, weights[:, i], label=f"w{i}")
plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.title("Weights over Epochs")
plt.grid()
plt.legend()
plt.show()

# 3. Bias over time
plt.figure(figsize=(6, 4))
plt.plot(epochs, biases, label="Bias b", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Bias")
plt.title("Bias over Epochs")
plt.grid()
plt.legend()
plt.show()

# 4. Gradient Magnitudes
grad_w = np.array(grad_w)
plt.figure(figsize=(10, 4))
for i in range(grad_w.shape[1]):
    plt.plot(epochs, grad_w[:, i], label=f"dL/dw{i}")
plt.xlabel("Epoch")
plt.ylabel("Gradient Value")
plt.title("Weight Gradients over Epochs")
plt.grid()
plt.legend()
plt.show()

# 5. Predictions vs True
plt.figure(figsize=(10, 4))
for i in range(X.shape[0]):
    pred_i = [p[i] for p in predictions_per_epoch]
    plt.plot(epochs, pred_i, label=f"Pred sample {i}")
plt.hlines(y_batch, xmin=0, xmax=num_epochs-1, colors='gray', linestyles='dashed', label="True y")
plt.xlabel("Epoch")
plt.ylabel("Predicted Value")
plt.title("Predictions over Epochs")
plt.grid()
plt.legend()
plt.show()
