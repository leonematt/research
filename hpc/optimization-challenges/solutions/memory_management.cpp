/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include <chrono>

using namespace std;

/*

   The solution for this challenge is to move the memory allocation out of the loop.
   High performance and high frequency trading code will allocate memory before
   it enters the hotpath, to minimize unnecessary memory allocation.

*/

int main(int argc, char **argv)
{

    int n = 1000000;

    int *nums = new int[10];

    auto start = chrono::high_resolution_clock::now();
    // BEGIN HOTPATH CODE

    for(int i = 0; i < 1000000; i++) {}

    // END HOTPATH CODE
    auto stop = chrono::high_resolution_clock::now();

    delete nums;

    auto totalTime = chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "The hotpath took a total time of: " << totalTime.count() << " microseconds!" << endl;

    cout << "By not allocating the memory in the hot path, we save precious time doing the same work!" << endl;

    return 0;

}
