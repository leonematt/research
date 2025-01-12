/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef HARDWARE_H
#define HARDWARE_H

#include <string>

#include <iostream>

#include <vector>

#include <memory>

using namespace std;

class HardwareDevice
{

public:

    virtual void build() {}

};

class XEONCPU : public HardwareDevice
{

public:

    string name;

    XEONCPU() {}
    ~XEONCPU() {}

    void build() override
    {

        cout << "We built a XEON CPU" << endl;

    }

};

class CoreCPU : public HardwareDevice
{

public:

    string name;

    CoreCPU() {}
    ~CoreCPU() {}

    void build() override
    {

        cout << "We built a Core CPU" << endl;

    }

};

class E810NIC : public HardwareDevice
{

public:

    string name;

    E810NIC() {}
    ~E810NIC() {}

    void build() override
    {

        cout << "We built an E810 NIC" << endl;

    }

};

class IPUNIC : public HardwareDevice
{

public:

    string name;

    IPUNIC() {}
    ~IPUNIC() {}

    void build() override
    {

        cout << "We built an IPU NIC" << endl;

    }

};

class HardwareGroup
{

public:

    vector<shared_ptr<HardwareDevice>> deviceVector;

    HardwareGroup() {}
    ~HardwareGroup() {}

    void build()
    {

        for (shared_ptr<HardwareDevice> i : deviceVector)
            i->build();

        deviceVector.clear();

    }

    void addDevice(shared_ptr<HardwareDevice> device)
    {
        deviceVector.push_back(device);
    }

    void removeDevice()
    {
        deviceVector.pop_back();
    }

};

shared_ptr<HardwareDevice> getHardwareDevice(string kind)
{

    shared_ptr<HardwareDevice> ptr = NULL;

    if (kind.compare("XEON") == 0)
        ptr = dynamic_pointer_cast<HardwareDevice>(shared_ptr<XEONCPU>(new XEONCPU()));

    else if (kind.compare("Core") == 0)
        ptr = dynamic_pointer_cast<HardwareDevice>(shared_ptr<CoreCPU>(new CoreCPU()));

    if (kind.compare("E810") == 0)
        ptr = dynamic_pointer_cast<HardwareDevice>(shared_ptr<E810NIC>(new E810NIC()));

    else if (kind.compare("IPU") == 0)
        ptr = dynamic_pointer_cast<HardwareDevice>(shared_ptr<IPUNIC>(new IPUNIC()));

    return ptr;

}

#endif // HARDWARE_H