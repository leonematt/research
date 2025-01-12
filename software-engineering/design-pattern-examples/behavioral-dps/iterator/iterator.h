/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef ITERATOR_H
#define ITERATOR_H

#include <memory>

using namespace std;

template <typename T>

class Iterator
{

public:

    unique_ptr<T[]> currentCollection;

    int currentPosition;

    Iterator(unique_ptr<T[]> integer)
    {

        currentCollection = move(integer);

        currentPosition = 0;

    }

    Iterator(const Iterator& other){}

    ~Iterator() {}

    Iterator operator++(int)
    {

        Iterator temp = *this;

        currentPosition++;

        return temp;

    }

    Iterator operator--(int)
    {

        Iterator temp = *this;

        currentPosition--;

        return temp;

    }

    int operator*()
    {

        return currentCollection[currentPosition];

    }


};

#endif // ITERATOR_H