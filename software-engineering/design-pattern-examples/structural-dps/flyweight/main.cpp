/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include "function.h"

using namespace std;

int main(void)
{

    // Set up the first lambda function
    shared_ptr<Function> function1 = make_shared<Function>();

    // Set up the second lambda function
    shared_ptr<Function> function2 = function1->cloneFunction();

    // Dump location of each Function
    cout << "Function 1 is at: " << function1 << endl;
    cout << "Function 2 is at: " << function2 << endl << endl;

    // Dump location of each Function's context
    cout << "Function 1's Context is at: " << function1->getContext() << endl;
    cout << "Function 2's Context is at: " << function2->getContext() << endl;

    return 0;

}