#
# Copyright (C) 2023 Matthew Leon
# SPDX-License-Identifier: BSD-3-Clause
#

import sys

import numpy as np


# Generate stochastic volatility paths
def generate_heston_paths(isp, iv, total_time, num_steps, num_paths, rho, kappa, theta, sigma, r):

    # Set numpy seed
    np.random.seed(42)

    # Time steps for each path
    dt = total_time / num_steps

    # Initialize stochastic volatility and price paths

    # Stochastic volatility paths
    svp = np.zeros((num_steps, num_paths))

    # Stock price paths
    spp = np.zeros((num_steps, num_paths))

    # Set intial volatility and stock prices
    svp[0] = iv
    spp[0] = isp

    # Generate each path
    for i in range(1, num_steps):

        # Generate random data for the Weiner processes
        dw1 = np.random.normal(size=num_paths) * np.sqrt(dt)
        dw2 = rho * dw1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=num_paths) * np.sqrt(dt)

        # Calculate next volatility path
        svp[i] = np.abs(svp[i - 1] + kappa * (theta - svp[i - 1]) * dt + sigma * np.sqrt(svp[i - 1]) * dw2)

        # Calculate the next stock price path
        spp[i] = spp[i-1] * np.exp((r - (svp[i - 1] / 2)) * dt + np.sqrt(svp[i - 1]) * dw2)

    return spp

def main():

    # Initial Stock Price
    isp = 100

    # Initial Volatility
    iv = 0.1

    # Total amount of time in years
    time = 1

    # Number of time steps
    steps = 100

    # Number of paths
    paths = 1000

    # Correlation Coefficient
    rho = 0.6

    # Reversion Rate
    kappa = 1

    # Long term volatility
    theta = 0.02

    # Volatility of the Volatility
    sigma = 0.4

    # Risk-Free Return Rate
    r = 0.01

    # Generate the Stock Price Paths
    stock_price_paths = generate_heston_paths(isp=isp, iv=iv, total_time=time, num_steps=steps, num_paths=paths, 
                                              rho=rho, kappa=kappa, theta=theta, sigma=sigma, r=r)

    np.set_printoptions(threshold=np.inf)

    data_str = np.array2string(stock_price_paths)

    # Write the data to a txt file
    with open(sys.argv[1], "w") as fdata:
        fdata.write(data_str)

if __name__ == "__main__":

    if (len(sys.argv) != 2):
        print("Usage:")
        print("python3 gen_heston_data.py <filename>")

    main();