/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "tradingstrategy.h"

int main(void)
{

    // Create the TradingSystem
    TradingSystem TS("HFT");

    // Print the type of strategy
    TS.getStrategyType();

    // Switch the type of strategy
    TS.switchStrategy("MFT");

    // Print the type of strategy
    TS.getStrategyType();

    return 0;

}