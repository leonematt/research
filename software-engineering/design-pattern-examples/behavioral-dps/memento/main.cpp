/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <vector>
#include <iostream>

#include "counter.h"

using namespace std;

int main(void)
{

    // Allocate a Counter
    Counter myCounter(0);


    // Increment the Counter
    myCounter.increment();

    // Undo the increment
    myCounter.undo();

    // Try to undo an increment that is at its initial state
    myCounter.undo();

    return 0;

}