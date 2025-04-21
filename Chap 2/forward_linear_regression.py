import numpy as np
from typing import Dict, Tuple, Union
from numpy import ndarray

def forward_Linear_Regression (X_batch: ndarray, y_batch: ndarray, weights: Dict [str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:


    # Assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # Assert that matrix multiplication can work
    assert X_batch.shape[1] == weights ['W'].shape[0]

    # Assert that B (bias) is a 1*1
    assert weights['B'].shape[0] ==  weights['B'].shape[1] == 1

    # Compute the operations on the forward pass

    N =np.dot(X_batch, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    # Save the info 

    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

