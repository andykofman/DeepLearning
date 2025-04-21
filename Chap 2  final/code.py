import numpy as np
from numpy import ndarray

from typing import Callable, Dict, Tuple, List

#Lower percision for higher monitoring
np.set_printoptions(precision=4)

# %load_ext autoreload
# %autoreload 2

TEST_ALL = False

# Boston data
from sklearn.datasets import load_boston
boston = load_boston()
data = boston.data
target = boston.target
features = boston.features_names

"""
Scikit Learn Linear Regression
Step 1: Data prep
"""
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target,test_size = 0.3, random_state = 80718)
y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)