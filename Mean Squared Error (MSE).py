import numpy as np

# Step 1: Generate sample data
X = np.array ([
    [2.0, 3.5, 1.1],        # 3 features
    [1.0, 2.0, 3.0],
    [4.0, 1.5, 2.2],
    [3.0, 2.5, 1.8]
    # 4 samples
])

# True Target values (y_batch)
y_batch = np.array ([1.0,0.0,1.0,0.0])

# Step 2: Initialize weights and bias (parameters)

w = np.array([0.5,-0.3,0.8])
b = 1.0

# Step 3: Make predictions

def predict (X, w, b):
    return np.dot(X,w) + b

# Step 4: Calculate Mean Squared Error

def calculate_mse(p_batch, y_batch): # p_batch --> predictions, y_batch --> true values
    return np.mean((p_batch - y_batch) ** 2)

# Step 5: Training Loop
learning_rate = 0.01
num_epochs = 500 

for epoch in range (num_epochs):
    # Forward Pass
    p_batch = predict(X, w, b)

    # Calculate Loss
    loss = calculate_mse(p_batch, y_batch)

    # Calculate Gradients
    dw = -2 * np.dot(X.T, (y_batch - p_batch)) / len(X)
    db = -2 * np.mean(y_batch - p_batch)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # print and debug each 10 epochs
    if epoch % 10 == 0 :
        print (f"Epoch {epoch}, Loss:{loss: .4f} " )    

final_predictions = predict(X,w,b)
print("\nFinal Predictions:", final_predictions)
print("True Values:", y_batch)
print("Final Loss:", calculate_mse(y_batch, final_predictions))