/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include "client.h"
#include "adapter.h"

using namespace std;

int main(void)
{
    // Create the Target
    shared_ptr<Target> originalTarget = make_shared<Target>();

    // Create the Target Adapter
    shared_ptr<TargetAdapter> targetAdapter = make_shared<TargetAdapter>(originalTarget);

    // Create the Client
    shared_ptr<Client> client = make_shared<Client>("1337");

    // Send the data from the client to the target using the adapter
    client->transferDataToTarget(targetAdapter);

    // Print out the data from the Target to make sure it was received
    cout << "DATA RECEIVED BY TARGET: " << originalTarget->getData() << endl;

    return 0;

}