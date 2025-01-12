/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "portfolio.h"

#include "analyzer.h"

void Portfolio::setRiskAnalyzer()
{

    riskAnalyzer = make_shared<RiskAnalyzer>();

}

void HFTPortfolio::calculateRisk()
{

    riskAnalyzer->analyze((HFTPortfolio &)*this);

}

void MFTPortfolio::calculateRisk()
{

    riskAnalyzer->analyze((MFTPortfolio &)*this);

}

void LFTPortfolio::calculateRisk()
{

    riskAnalyzer->analyze((LFTPortfolio &)*this);

}