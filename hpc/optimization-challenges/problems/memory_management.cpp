/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include <chrono>

using namespace std;

/*

    In this code, we need to allocate the nums memory, but the way this is done is quite inefficient.
    The goal is to find a way to allocate the memory more efficiently than the current solution provides.

*/

int main(int argc, char **argv)
{

    int n = 1000000;

    auto start = chrono::high_resolution_clock::now();
    // BEGIN HOTPATH CODE

    for(int i = 0; i < 1000000; i++)
    {
        int *nums = new int[10];
        delete nums;
    }

    auto stop = chrono::high_resolution_clock::now();
    // END HOTPATH CODE

    auto totalTime = chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "The hotpath took a total time of: " << totalTime.count() << " microseconds!" << endl;

    return 0;

}
