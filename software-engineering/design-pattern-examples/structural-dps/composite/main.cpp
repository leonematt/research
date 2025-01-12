/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "hardware.h"

int main(void)
{

    // Allocate Hardware Group
    shared_ptr<HardwareGroup> myGroup = make_shared<HardwareGroup>();

    // Allocate Hardware Devices
    myGroup->addDevice(getHardwareDevice("XEON"));
    myGroup->addDevice(getHardwareDevice("Core"));
    myGroup->addDevice(getHardwareDevice("E810"));
    myGroup->addDevice(getHardwareDevice("IPU"));

    // Build the Hardware Devices
    myGroup->build();

    return 0;

}