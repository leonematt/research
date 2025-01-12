/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include <chrono>

using namespace std;

/*

    In this code, we need to find the summation of a matrix.
    Unfortunately the code is not cache-optimized, so find a way to make it more cache-efficient.
    We still need to sum the matrix values.
*/


int main(int argc, char **argv)
{

    int n = 1000;

    int matrix[n][n];

    auto start = chrono::high_resolution_clock::now();
    // BEGIN HOTPATH CODE

    // Populate matrix
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            matrix[i][j] = i * j;

    int sum = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum += matrix[i][j];

    // END HOTPATH CODE
    auto stop = chrono::high_resolution_clock::now();

    auto totalTime = chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "The hotpath took a total time of: " << totalTime.count() << " microseconds!" << endl;

    return 0;

}
