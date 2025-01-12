/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef CLIENT_H
#define CLIENT_H

#include <string>

#include <memory>

#include "adapter.h"

using namespace std;

class Client
{

public:

    string clientValue;

    Client(string value)
    {

        clientValue = value;

    }
    ~Client() {}

    void transferDataToTarget(shared_ptr<TargetAdapter> adapter)

    {

        cout << "SENDING DATA: " << clientValue << endl;

        adapter->setTargetValue(clientValue);

    }

};

#endif // CLIENT_H