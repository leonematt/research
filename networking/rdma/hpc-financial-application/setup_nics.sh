#!/bin/bash

#
# Copyright (C) 2023 Matthew Leon
# SPDX-License-Identifier: BSD-3-Clause
#

# Load the irdma driver
sudo modprobe irdma

# Bring up both network cards
sudo ip link set enp1s0f0 up
sudo ip link set enp2s0f0 up

# Configure both network cards ip address
sudo ifconfig enp1s0f0 10.0.1.5
sudo ifconfig enp2s0f0 10.0.1.6

# Add both cards to rdma link using soft-iwarp
sudo rdma link add siwdev1 type siw netdev enp1s0f0
sudo rdma link add siwdev2 type siw netdev enp2s0f0
