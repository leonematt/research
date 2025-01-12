/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>

#include "tsmediator.h"

using namespace std;

#ifndef TRADINGENGINE_H
#define TRADINGENGINE_H

class TradingEngine
{

public:

    shared_ptr<TSMediator> mediator;

    TradingEngine() {}
    ~TradingEngine() {}

    void setMediator(shared_ptr<TSMediator> m)
    {

        mediator = m;

    }

    void process(string Ticker, int numShares, bool shouldBuy)
    {

        mediator->submitOrder(Ticker, numShares, shouldBuy);

    }

};

#endif // TRADINGENGINE_H