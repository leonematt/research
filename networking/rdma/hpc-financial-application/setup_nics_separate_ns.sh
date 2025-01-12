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

# Create and each network card to namespace
sudo ip netns add net0
sudo ip netns add net1

sudo ip link set dev enp1s0f0 netns net0
sudo ip link set dev enp2s0f0 netns net1

# Configure both network cards ip address
sudo ip netns exec net0 ifconfig enp1s0f0 10.0.1.5
sudo ip netns exec net1 ifconfig enp2s0f0 10.0.1.6

# Set up routes for each card to each other
sudo ip netns exec net0 ip route add 10.0.1.6 via 10.0.1.5
sudo ip netns exec net1 ip route add 10.0.1.5 via 10.0.1.6

# Add both cards to rdma link using soft-iwarp
sudo ip netns exec net0 rdma link add siwdev1 type siw netdev enp1s0f0
sudo ip netns exec net1 rdma link add siwdev2 type siw netdev enp2s0f0
