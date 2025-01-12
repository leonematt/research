'''
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
'''

# Import required modules
import numpy as np

from sklearn.datasets import make_regression
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set numpy random seed
seed = 42
np.random.seed(seed)

# Generate the data
X, Y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=seed)

# Split the data so we can fit and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Initialize and fit the model to our generated data
# Neural Network with 3 hidden layers
MLPmodel = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
MLPmodel.compile(optimizer='adam',
              loss='mean_squared_error')

# Train the model
MLPmodel.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Get the accuracy
mean_squared_error = MLPmodel.evaluate(X_test, y_test, verbose=0)

print(f"The Model's Mean Squared Error is: {mean_squared_error}")
