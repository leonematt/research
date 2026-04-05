'''
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
'''

# Import required modules
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set numpy random seed
seed = 42
np.random.seed(seed)

# Generate the data
X, Y = make_regression(n_samples=10000, n_features=3, noise=0.15, random_state=seed)
polynomial_feat = PolynomialFeatures(degree=2, include_bias=False)
x_poly = polynomial_feat.fit_transform(X)
Y = Y + np.sum(x_poly[:, 1:], axis=1)

# Split the data so we can fit and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Initialize and fit the model to our generated data
# Model with L2 regularization penalty
RModel = Ridge(alpha=0.1)
RModel.fit(X_train, y_train)

# Test the model
y_prediction = RModel.predict(X_test)

# Get the accuracy
mse = mean_squared_error(y_test, y_prediction)

print(f"The Model's Mean Squared Error is: {mse}")
