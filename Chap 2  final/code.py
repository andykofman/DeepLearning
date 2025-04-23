
# %%

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
# Add polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(data)


# Then use this transformed data in your train_test_split
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

# Random Forest often performs better on housing data
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=80718)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

# Visualization
import matplotlib.pyplot as plt

# Sample a subset for clearer plots
sample_size = 300
if X_test.shape[0] > sample_size:
    idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sample = X_test[idx]
    y_test_sample = y_test[idx]
    preds_sample = preds[idx]
else:
    X_test_sample = X_test
    y_test_sample = y_test
    preds_sample = preds
# Plot: Predicted vs. Actual values
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs. Actual values for\nLinear Regression model")
plt.xlim([0, max(preds_sample.max(), y_test_sample.max())])
plt.ylim([0, max(preds_sample.max(), y_test_sample.max())])
plt.scatter(preds_sample, y_test_sample)
plt.plot([0, max(preds_sample.max(), y_test_sample.max())], [0, max(preds_sample.max(), y_test_sample.max())])
plt.show()

# Plot: Relationship between a feature and target
plt.scatter(X_test_sample[:, 0], y_test_sample)  # 0 = median income in California housing
plt.xlabel("Median Income (scaled)")
plt.ylabel("Target (Median House Value)")
plt.title("Relationship between Median Income and Target")
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
print(f"R² Score: {r2_score(y_test, preds):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")




# %%
