"""
Feature scaling in python (Standardization and Normalization)

In this section I work with concrete compressive strength dataset.
The regression problem is predicting concrete compressive strength, given quantities of seven components and the age of the concrete. 
There are 8 numerical input variables and 1030 instances in this dataset.


https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
"""
from ucimlrepo import fetch_ucirepo 
import pandas as pd  # Add this import
  
# fetch dataset 
concrete_compressive_strength = fetch_ucirepo(id=165) 
  
# data (as pandas dataframes) 
X = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 
  
# # metadata 
# print(concrete_compressive_strength.metadata) 
  
# # variable information 
# print(concrete_compressive_strength.variables) 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# First: Normalization
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler().fit(X_train)  # Changed from X_test to X_train
X_train_norm = min_max_scaler.transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)


# Second: Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
# fit on training data
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#Visualization od the scalling effects

import matplotlib.pyplot as plt
import seaborn as sns

# Select a few features to compare
features = X.columns[:3] # The first five features


# Create subplots for each feature
fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Changed to 3x3 grid since we have 3 plots per feature
fig.suptitle('Distribution of Features: Original vs Normalized vs Standardized')

# Modify the visualization section
for i, feature in enumerate(features):
    # original data
    sns.histplot(X_train[feature], ax=axes[i, 0], kde=True)
    axes[i, 0].set_title(f'Original: {feature}')
    axes[i, 0].axvline(X_train[feature].mean(), color='r', linestyle='--', label='Mean')
    axes[i, 0].legend()

    # Normalized
    sns.histplot(X_train_norm[:, i], ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'Normalized: {feature}')
    axes[i, 1].axvline(X_train_norm[:, i].mean(), color='r', linestyle='--', label='Mean')
    axes[i, 1].legend()
    
    # Standardized data
    sns.histplot(X_train_std[:, i], ax=axes[i, 2], kde=True)
    axes[i, 2].set_title(f'Standardized: {feature}')
    axes[i, 2].axvline(0, color='r', linestyle='--', label='Mean')
    axes[i, 2].legend()

plt.tight_layout()
plt.show()

# Basic statistics comparison
print("\nOriginal Data Statistics:")
print(X_train.describe())
print("\nNormalized Data Statistics:")
print(pd.DataFrame(X_train_norm, columns=X.columns).describe())
print("\nStandardized Data Statistics:")
print(pd.DataFrame(X_train_std, columns=X.columns).describe())

# Add this after the standardization section
print("\nVerifying Standardization:")
std_means = pd.DataFrame(X_train_std).mean()
std_stds = pd.DataFrame(X_train_std).std()

print("\nMeans of standardized features:")
for feature, mean in zip(X.columns, std_means):
    print(f"{feature}: {mean:.6f}")

print("\nStandard deviations of standardized features:")
for feature, std in zip(X.columns, std_stds):
    print(f"{feature}: {std:.6f}")