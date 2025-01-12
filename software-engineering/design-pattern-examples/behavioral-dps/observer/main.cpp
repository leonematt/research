/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "kernel.h"

using namespace std;

int main(void)
{


    // Allocate a Kernel
    Kernel myKernel = Kernel();

    // Allocate a new driver and register it
    myKernel.registerDriver(make_shared<DeviceDriver>());

    // Switch Kernel to Low Power Mode
    myKernel.switchToLowPowerMode();

    return 0;

}