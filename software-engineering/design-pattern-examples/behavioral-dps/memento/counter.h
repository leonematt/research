/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef COUNTER_H
#define COUNTER_H

#include <stack>

#include <iostream>

using namespace std;

class Counter
{

public:

    Counter(int value = 0) 
    {

        counter = value;

    }

    void increment()
    {

        counter++;
        counterOperations.push(true);
        cout << "Counter value is now: " << counter << endl;

    }

    void decrement()
    {

        counter--;
        counterOperations.push(false);
        cout << "Counter value is now: " << counter << endl;

    }

    void undo()
    {

        if (counterOperations.empty()) {

            cout << "Nothing to undo" << endl;
            return;

        }

        bool operation = counterOperations.top();
        counterOperations.pop();

        if (operation) {

            cout << "Undoing increment" << endl;
            counter--;

        }

        else {

            cout << "Undoing decrement" << endl;
            counter--;

        }

        cout << "Counter value is now: " << counter << endl;

    }

private:

    int counter;

    stack<bool> counterOperations;

};

#endif // COUNTER_H