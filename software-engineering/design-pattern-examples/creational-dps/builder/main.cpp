/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "computer.h"

int main(void)
{

    // Set up the ComputerStore
    shared_ptr<ComputerStore> myIntelCB = make_shared<ComputerStore>();

    // Get a Computer
    shared_ptr<Computer> myNewPC = myIntelCB->getComputer("Intel");
    if (myNewPC)
        cout << "We got a new PC!" << endl;

    return 0;

}

