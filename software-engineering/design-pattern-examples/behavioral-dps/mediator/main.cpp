/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "tsmediator.h"
#include "orderengine.h"
#include "tradingengine.h"

using namespace std;

int main(void)
{

    // Set up order engine
    shared_ptr<OrderEngine> oEngine = make_shared<OrderEngine>();

    // Set up trading engine
    shared_ptr<TradingEngine> tEngine = make_shared<TradingEngine>();

    // Set up mediator
    shared_ptr<TSMediator> mediator = make_shared<TSMediator>(tEngine, oEngine);

    // Set the mediator for each engine
    tEngine->setMediator(mediator);
    oEngine->setMediator(mediator);

    // Trading System places an order to buy 10 shares of $SPXL
    tEngine->process("SPXL", 10, true);

    return 0;

}