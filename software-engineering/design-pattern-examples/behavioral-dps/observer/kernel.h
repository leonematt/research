/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

#ifndef KERNEL_H
#define KERNEL_H

class DeviceDriver
{

public:

    DeviceDriver() {}
    ~DeviceDriver() {}

    void switchToLowPowerMode() {

        cout << "Driver is now in Low Power Mode" << endl;

    }

};

class Kernel
{

public:

    vector<shared_ptr<DeviceDriver>> registeredDrivers;

    Kernel() {}
    ~Kernel()
    {

        registeredDrivers.clear();

    }

    void registerDriver(shared_ptr<DeviceDriver> driver)
    {

        cout << "Registering Driver" << endl;

        registeredDrivers.push_back(driver);

    }


    void switchToLowPowerMode()
    {

        cout << "Switching Kernel and Drivers to Low Power Mode" << endl;

        for (auto i : registeredDrivers)
            i->switchToLowPowerMode();

    }

};

#endif // KERNEL_H