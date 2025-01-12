/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include <chrono>

using namespace std;

/*

    The solution for this code is to change the way the matrix
    is accessed. In the un-optimized code, we access the data in
    a column-by-column manner, which cannot be easily optimized.

    By switching to a row-by-row access manner, we are bringing in
    the next data point in by the cache and that makes it much faster.

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
            matrix[j][i] = i * j;

    int sum = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum += matrix[j][i];
    
    // END HOTPATH CODE
    auto stop = chrono::high_resolution_clock::now();

    auto totalTime = chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "The hotpath took a total time of: " << totalTime.count() << " microseconds!" << endl;

    return 0;

}
