/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <iostream>

#include <memory>

using namespace std;

#ifndef PORTFOLIO_H
#define PORTFOLIO_H

class Analyzer;
class RiskAnalyzer;

class Portfolio
{

public:

    shared_ptr<RiskAnalyzer> riskAnalyzer;

    virtual ~Portfolio() {}

    void setRiskAnalyzer();

    virtual void calculateRisk() {}

};

class HFTPortfolio : public Portfolio
{

public:

    void calculateRisk() override;

};

class MFTPortfolio : public Portfolio
{

public:

    void calculateRisk() override;

};

class LFTPortfolio : public Portfolio
{

public:

    void calculateRisk() override;

};


#endif // PORTFOLIO_H