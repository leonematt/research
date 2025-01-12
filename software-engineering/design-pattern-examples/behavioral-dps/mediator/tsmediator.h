/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>

using namespace std;

class TradingEngine;
class OrderEngine;

#ifndef TSMEDIATOR_H
#define TSMEDIATOR_H

class TSMediator
{

public:

    shared_ptr<TradingEngine> tEngine;
    shared_ptr<OrderEngine> oEngine;

    TSMediator(shared_ptr<TradingEngine> t, shared_ptr<OrderEngine> o) 
    {

        tEngine = t;

        oEngine = o;

    }
    ~TSMediator() {}

    void submitOrder(string Ticker, int numShares, bool shouldBuy);

};

#endif // TSMEDIATOR_H