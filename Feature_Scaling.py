"""
Feature scaling in python (Standardization and Normalization)

In this section I work with concrete compressive strength dataset.
The regression problem is predicting concrete compressive strength, given quantities of seven components and the age of the concrete. 
There are 8 numerical input variables and 1030 instances in this dataset.


https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
"""
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
concrete_compressive_strength = fetch_ucirepo(id=165) 
  
# data (as pandas dataframes) 
X = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 
  
# metadata 
print(concrete_compressive_strength.metadata) 
  
# variable information 
print(concrete_compressive_strength.variables) 

# import pandas as pd
# # First: Normalization
# from sklearn.preprocessing import MinMaxScaler

# from sklearn.preprocessing import StandardScaler

# min_max_scaler = MinMaxScaler().fit(X_test)
# X_norm = min_max_scaler.transform(X)

# # Second: Standardization
# scaler = StandardScaler().fit(X_train)
# X_std = scaler.transform(X)
