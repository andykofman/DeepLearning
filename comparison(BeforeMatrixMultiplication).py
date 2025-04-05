#Inputs: 1 training Example with 3 features

x1, x2, x3 = 2.0 , 3.5 , 1.1
w1, w2, w3 = 0.5, -0.3 , 0.8
b = 1.0

y_hat = w1 * x1 + w2 * x2 + w3 * x3 + b


# Inputs : 4 training examples (samples) with 4 features each

X = [
    [x11, x12, x13],
    [x21, x22, x23],
    [x31, x32, x33],
    [x41, x42, x43]


]

#weights
w = [w1, w2, w3]

b= 1.0

predictions = []

for sample in X :
    y_hat = sample [0]*w[0] + sample [1]*w[1] + sample [2] *w[2] +b
    predictions.append(y_hat)

# Verbose

# Harder to scale

# Slower for large datasets

# Not optimized for GPU/parallel computation

# Painful for computing gradients during training



####Matrix Multiplication for the rescue:
import nmpy as np


X = np.arrat([[x11, x12, x13],
    [x21, x22, x23],
    [x31, x32, x33],
    [x41, x42, x43]])

w =np.array([w1,w2,w3])

b = 1.0 

y_hat = X.dot(w) +b

# This version is:
#     Cleaner

#     Faster

#     Easier to debug

#     Compatible with deep learning frameworks

#     Ready for gradient-based optimization