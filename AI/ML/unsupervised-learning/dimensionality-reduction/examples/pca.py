'''
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
'''

# Import required modules
import numpy as np

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA

# Set numpy random seed
seed = 42
np.random.seed(seed)

# Generate the data
X, Y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=seed)

# Initialize and fit the model to our generated data
# This model condenses the data down into minimal components
nc = 4
PCAModel = PCA(n_components=nc)
PCAModel.fit_transform(X)

# Dump the explained variance for each data
ev = PCAModel.explained_variance_ratio_
ev_sum = 0.0
for i, var in enumerate(PCAModel.explained_variance_ratio_):
    ev_sum = ev_sum + var
    print(f"Explained variance for Principal Component {i}: {var * 100:.2f}%")

print(f"Total Explained variance: {ev_sum * 100:.2f}%")