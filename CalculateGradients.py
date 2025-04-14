
import numpy as np
def loss_gradients (forward_info: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """
    Computes the gradients (dLdW and dLdB) for the step-by-step linear regression model.
    """

    batch_size = forward_info['X'].shape[0]
    dLdP = -2 * (forward_info['Y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dNdW = np.transpose(forward_info['X'], (1,0))
  
    dLdB = np.ones_like(forward_info['N'])

    dLdN = dLdP * dPdN
    dLdW = np.dot(dNdW, dLdN)

    # need to sum along dimension representing the batch size
    # (see note near the end of this chapter)
    dLdB = (dLdP * dPdB).sum(axis=0)
    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    return loss_gradients

# forward_info, loss = forward_loss(X_batch, y_batch, weights)
# loss_grads = loss_gradients(forward_info, weights)
# for key in weights.keys():
#     weights[key] -= learning_rate * loss_grads[key]

# #run the train function for a certain number of epochs
# train_info = train(X_train, y_train, learning_rate =0.001, batch_size = 3, return_weights = True, seed =80718)
