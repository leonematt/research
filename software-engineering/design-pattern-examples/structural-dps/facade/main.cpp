/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "broker.h"

int main(void)
{

    // Allocate the Broker
    shared_ptr<Broker> myBroker = make_shared<Broker>();

    // Buy 100 shares of DPST and TQQQ
    myBroker->buySecurity("DPST", 100);
    myBroker->buySecurity("TQQQ", 100);

    // Sell 1 call option for TQQQ
    myBroker->sellSecurity("TQQQ-CALL", 1);

    // Buy 1 US Treasury Bond
    myBroker->buySecurity("912797FW2", 1);

    // Exchange USD for 10000 GBP
    myBroker->buySecurity("USD-GBP", 10000);

    myBroker.reset();

    return 0;

}