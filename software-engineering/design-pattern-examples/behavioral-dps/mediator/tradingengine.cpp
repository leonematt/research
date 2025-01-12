/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "tsmediator.h"
#include "tradingengine.h"
#include "orderengine.h"

void TSMediator::submitOrder(string Ticker, int numShares, bool shouldBuy)
{

    oEngine->placeOrder(Ticker, numShares, shouldBuy);

}