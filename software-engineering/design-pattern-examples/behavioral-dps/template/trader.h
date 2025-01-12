/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

using namespace std;

#ifndef TRADER_H
#define TRADER_H

class SecuritiesTrader
{

public:

    void logTrade(string ticker, int numShares)
    {

        cout << "Logging trade for " << numShares << " of " << ticker << endl;

    }

    virtual void placeOrder(string ticker, int numShares) {}

    // Template Method
    void trade(string ticker, int numShares)
    {
        // Place the order and log the trade
        placeOrder(ticker, numShares);
        logTrade(ticker, numShares);

    }

};

class StockTrader : public SecuritiesTrader
{

public:

    void placeOrder(string ticker, int numShares) override
    {

        cout << "Placing order for " << numShares << " of " << ticker << endl;

    }

};

class OptionsTrader : public SecuritiesTrader
{

public:

    void placeOrder(string ticker, int numShares) override
    {

        cout << "Placing order for " << numShares << " of " << ticker << endl;

    }

};

#endif // TRADER_H