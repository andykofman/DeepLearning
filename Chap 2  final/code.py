import numpy as np
from numpy import ndarray

from typing import Callable, Dict, Tuple, List

#Lower percision for higher monitoring
np.set_printoptions(precision=4)

# %load_ext autoreload
# %autoreload 2

TEST_ALL = False

# California housing data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data = housing.data
target = housing.target
features = housing.feature_names

"""
Scikit Learn Linear Regression
Step 1: Data prep
"""
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target,test_size = 0.3, random_state = 80718)
#The -1 in reshape means "automatically calculate this dimension" 
# while 1 ensures we have a 2D array with one column
# Reshape y values to 2D arrays
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

"""
Step 2: Linear Regression
"""
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train,y_train)
preds = lr.predict (X_test)

# Visualization
# Visualization
import matplotlib.pyplot as plt
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs. Actual values for\nLinear Regression model")
max_val = max(max(preds), max(y_test))
plt.xlim([0, max_val])
plt.ylim([0, max_val])
plt.scatter(preds, y_test)
plt.plot([0, max_val], [0, max_val])
plt.show()