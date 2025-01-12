/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <string>
#include <iostream>

using namespace std;

#ifndef TRADINGSTRATEGY_H
#define TRADINGSTRATEGY_H

class TradingStrategy
{

public:

    virtual ~TradingStrategy() {};

};

class HFTStrategy : public TradingStrategy
{

public:

};

class MFTStrategy : public TradingStrategy
{

public:

};

class LFTStrategy : public TradingStrategy
{

public:

};

class TradingSystem
{

public:

    shared_ptr<TradingStrategy> strat;

    TradingSystem(string stype)
    {

        if (stype.compare("HFT") == 0)
            strat = make_shared<HFTStrategy>();

        if (stype.compare("MFT") == 0)
            strat = make_shared<MFTStrategy>();

        if (stype.compare("LFT") == 0)
            strat = make_shared<LFTStrategy>();

    }
    ~TradingSystem() {}

    void getStrategyType()
    {

        if(dynamic_pointer_cast<HFTStrategy>(strat))
            cout << "Strategy is HFT" << endl;

        if(dynamic_pointer_cast<MFTStrategy>(strat))
            cout << "Strategy is MFT" << endl;

        if(dynamic_pointer_cast<LFTStrategy>(strat))
            cout << "Strategy is LFT" << endl;

    }

    void switchStrategy(string stype)
    {

        if (stype.compare("HFT") == 0) {
            cout << "Switching to HFT Strategy" << endl;
            strat = make_shared<HFTStrategy>();
        }

        if (stype.compare("MFT") == 0) {
            cout << "Switching to MFT Strategy" << endl;
            strat = make_shared<MFTStrategy>();
        }

        if (stype.compare("HFT") == 0) {
            cout << "Switching to LFT Strategy" << endl;
            strat = make_shared<LFTStrategy>();
        }

    }

};

#endif // TRADINGSTRATEGY_H