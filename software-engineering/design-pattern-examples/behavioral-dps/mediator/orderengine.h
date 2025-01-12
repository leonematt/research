/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>
#include <iostream>

#include "tsmediator.h"

using namespace std;

#ifndef ORDERENGINE_H
#define ORDERENGINE_H

class OrderEngine
{

public:

    shared_ptr<TSMediator> mediator;

    OrderEngine() {}
    ~OrderEngine() {}

    void setMediator(shared_ptr<TSMediator> m)
    {

        mediator = m;

    }

    void placeOrder(string Ticker, int numShares, bool shouldBuy)
    {

        if (shouldBuy)
            cout << "Buying " << numShares << " shares of " << Ticker << endl;

        else
            cout << "Selling " << numShares << " shares of " << Ticker << endl;
    

    }

};

#endif // ORDERENGINE_H