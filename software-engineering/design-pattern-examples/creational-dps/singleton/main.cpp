/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "runtime.h"

#include <iostream>

using namespace std;

int main(void)
{

    shared_ptr<Runtime> cRuntime1 = Runtime::getRuntime();
    shared_ptr<Runtime> cRuntime2 = Runtime::getRuntime();

    cout << "The first  runtime points to the Runtime at: " << cRuntime1 << endl;
    cout << "The second runtime points to the Runtime at: " << cRuntime2 << endl;

    return 0;

}