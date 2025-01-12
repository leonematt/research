/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "trader.h"

int main(void)
{

    // Create a StockTrader
    StockTrader ST;

    // Create a OptionsTrader
    OptionsTrader OT;

    // Place trade for SPXL with StockTrader
    ST.trade("LABU", 5);

    // Place trade for LABU-CALL with OptionsTrader
    OT.trade("LABU-CALL", 5);

    return 0;

}