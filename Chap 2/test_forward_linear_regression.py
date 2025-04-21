import numpy as np
from typing import Dict, Tuple, Union
from numpy import ndarray
from forward_linear_regression import forward_Linear_Regression  # Fixed import statement

def test_forward_linear_regression():
    # Create sample test data
    X_batch = np.array([[1, 2],
                        [3, 4],
                        [5, 6]])  # 3 samples, 2 features each
    y_batch = np.array([[2], [4], [6]])  # 3 samples, 1 target value each
    
    # Create sample weights
    weights: Dict[str, ndarray] = {
        'W': np.array([[0.5], [0.5]]),  # 2x1 weight matrix
        'B': np.array([[0.5]])          # 1x1 bias matrix
    }
    
    # Run the forward pass
    loss, forward_info = forward_Linear_Regression(X_batch, y_batch, weights)
    
    # Print results
    print("Test Results:")
    print(f"Loss: {loss}")
    print(f"Forward Info:")
    print(f"- Input (X): \n{forward_info['X']}")
    print(f"- Intermediate (N): \n{forward_info['N']}")
    print(f"- Predictions (P): \n{forward_info['P']}")
    print(f"- Actual (y): \n{forward_info['y']}")

if __name__ == "__main__":
    test_forward_linear_regression()