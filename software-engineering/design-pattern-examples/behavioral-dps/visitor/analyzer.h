/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

using namespace std;

#ifndef ANALYZER_H
#define ANALYZER_H

#include "portfolio.h"

class Analyzer
{

public:

    Analyzer() {}
    ~Analyzer() {}

    virtual void analyze(HFTPortfolio &) {};
    virtual void analyze(MFTPortfolio &) {};
    virtual void analyze(LFTPortfolio &) {};

};

class RiskAnalyzer : public Analyzer
{

public:

    void analyze(HFTPortfolio &p) override
    {

        cout << "Calculating risk for High-Frequency-Trading Portfolio" << endl;

    }

    void analyze(MFTPortfolio &p) override
    {

        cout << "Calculating risk for Medium-Frequency-Trading Portfolio" << endl;
    }

    void analyze(LFTPortfolio &p) override
    {

        cout << "Calculating risk for Low-Frequency-Trading Portfolio" << endl;
    }

};

#endif // ANALYZER_H