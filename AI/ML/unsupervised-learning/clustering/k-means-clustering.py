'''
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
'''

# Import required modules
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Set numpy random seed
seed = 42
np.random.seed(seed)

# Generate the data
data, tlabels = make_blobs(n_samples=1000, centers=4, cluster_std=0.5, random_state=seed)

# Initialize and fit the model to our generated data
# This model condenses the data down into minimal components
nc = 4
KMEANSModel = KMeans(n_clusters=nc)
KMEANSModel.fit(data)

# Predict the data values
plabels = KMEANSModel.predict(data)

# Dump the adjusted rand index
ari = adjusted_rand_score(plabels, tlabels)
print(f"Adjusted Rand Index is: {ari:.2f}")