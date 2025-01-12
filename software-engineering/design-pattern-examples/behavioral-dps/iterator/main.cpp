/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <vector>
#include <iostream>

#include "iterator.h"

using namespace std;

int main(void)
{

    // Allocate an array of integers
    unique_ptr<int[]> myArray(new int[6]);

    // Populate the array
    for (int i = 0; i < 6; i++)
        myArray[i] = i + 1;

    // Create the iterator from the array
    Iterator<int> myIterator(move(myArray));

    // Loop through and print the value of each value in the array using the iterator
    for (int i = 0; i < 6; i++) {

        cout << *myIterator << endl;

        myIterator++;
    
    }

    return 0;

}