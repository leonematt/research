/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "portfolio.h"
#include "analyzer.h"

int main(void)
{

    // Allocate a High-Frequency-Trading Portfolio
    HFTPortfolio hftPortfolio;

    // Set up the Risk Analyzer
    hftPortfolio.setRiskAnalyzer();

    // Analyze the risk of the portfolio
    hftPortfolio.calculateRisk();

    // Allocate a Medium-Frequency-Trading Portfolio
    MFTPortfolio mftPortfolio;

    // Set up the Risk Analyzer
    mftPortfolio.setRiskAnalyzer();

    // Analyze the risk of the portfolio
    mftPortfolio.calculateRisk();

    // Allocate a Low-Frequency-Trading Portfolio
    LFTPortfolio lftPortfolio;

    // Set up the Risk Analyzer
    lftPortfolio.setRiskAnalyzer();

    // Analyze the risk of the portfolio
    lftPortfolio.calculateRisk();

    return 0;

}