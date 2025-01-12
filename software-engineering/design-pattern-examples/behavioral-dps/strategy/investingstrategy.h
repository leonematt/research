/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>
#include <iostream>

using namespace std;

#ifndef INVESTINGSTRATEGY_H
#define INVESTINGSTRATEGY_H

class InvestingStrategy
{

public:

    virtual ~InvestingStrategy() {};

};

class BAHStrategy : public InvestingStrategy
{

public:

};

class DividendStrategy : public InvestingStrategy
{

public:

};

class DCAStrategy : public InvestingStrategy
{

public:

};

class TradingSystem
{

public:

    shared_ptr<InvestingStrategy> strat;

    TradingSystem(){}
    ~TradingSystem() {}

    void getStrategyType()
    {

        if(dynamic_pointer_cast<BAHStrategy>(strat))
            cout << "Strategy is Buy-And-Hold" << endl;

        if(dynamic_pointer_cast<DividendStrategy>(strat))
            cout << "Strategy is Dividend" << endl;

        if(dynamic_pointer_cast<DCAStrategy>(strat))
            cout << "Strategy is Dollar-Cost-Averaging" << endl;

    }

    void setStrategy(shared_ptr<InvestingStrategy> strategy)
    {

        cout << "Setting the investing strategy" << endl;

        strat = strategy;

    }

};

#endif // INVESTINGSTRATEGY_H