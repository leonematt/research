/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "investingstrategy.h"

int main(void)
{

    // Create the TradingSystem
    TradingSystem TS;

    // Create a Buy-And-Hold Strategy
    shared_ptr<InvestingStrategy> s = make_shared<BAHStrategy>();

    // Create a Dollar-Cost-Averaging Strategy
    shared_ptr<InvestingStrategy> d = make_shared<DCAStrategy>();

    // Set the InvestingStrategy
    TS.setStrategy(s);

    // Get the InvestingStrategy
    TS.getStrategyType();

    // Set the InvestingStrategy
    TS.setStrategy(d);

    // Get the InvestingStrategy
    TS.getStrategyType();

    return 0;

}