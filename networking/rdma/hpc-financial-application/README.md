For installing and setting up rdma for e810, you need to do the following:

In your development directory go through the following steps:

1: Install your network card(s)
    The setup script assumes that you have 2 cards placed in the first and second pcie slots of your server

2: Build and Install irdma drivers

wget https://downloadmirror.intel.com/772536/irdma-1.1.11.tar.gz or get the latest irdma driver for your distro/os

tar -xvzf irdma-1.1.11.tar.gz

cd irdma-1.1.11

Meticulously follow the steps in README_irdma.txt


3: Reboot your machine

4: To get your cards recognized by rdma-core, run the setup_nics script

5: To verify your cards are recognized run: ibv_devices and you should see something like:
    device                 node GUID
    ------              ----------------
    siwdev1             527c6ffffe4f9b48
    siwdev2             527c6ffffe4fb7e0

    siwdev1 will have an IP address of 10.0.1.5
    siwdev2 will have an IP address of 10.0.1.6

6: To run rping test, run the following in 2 different terminals:

rping -sdvVa 10.0.1.6
rping -cdvVa 10.0.1.6 -C 10
