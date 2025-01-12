'''
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
'''

# Import required modules
import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set numpy random seed
seed = 42
np.random.seed(seed)

# Generate the data
X, Y = make_classification(n_samples=10000, n_features=10, n_informative=7, n_classes=3, random_state=seed)

# Split the data so we can fit and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and fit the model to our generated data
# Model that classifies using the nearest neighbors
KNNModel = KNeighborsClassifier(n_neighbors=5)
KNNModel.fit(X_train, y_train)

# Test the model
y_prediction = KNNModel.predict(X_test)

# Get the accuracy
accuracy_score = accuracy_score(y_test, y_prediction)

# Get the accuracy
print(f"The Model's Accuracy is: {accuracy_score * 100:.2f}%")
print(f"Model Report: {classification_report(y_test, y_prediction)}")
