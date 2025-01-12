/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef BROKER_H
#define BROKER_H

#include <string>
#include <iostream>
#include <memory>
#include <map>

using namespace std;

class SecurityConnection
{

public:

    virtual void buySecurity(string Ticker, uint numShares) {}
    virtual void sellSecurity(string Ticker, uint numShares) {}

};

class EquityConnection : public SecurityConnection
{

public:

    EquityConnection() {}
    ~EquityConnection() {}

    void buySecurity(string Ticker, uint numShares) override
    {

        cout << "You bought " << numShares << " shares of " << Ticker << "!" << endl;

    }

    void sellSecurity(string Ticker, uint numShares) override
    {

        cout << "You sold " << numShares << " shares of " << Ticker << "!" << endl;

    }

};

class DerivativeConnection : public SecurityConnection
{

public:

    DerivativeConnection() {}
    ~DerivativeConnection() {}

    void buySecurity(string Ticker, uint numShares) override
    {

        cout << "You bought " << numShares << " contracts of " << Ticker << "!" << endl;

    }

    void sellSecurity(string Ticker, uint numShares) override
    {

        cout << "You sold " << numShares << " contracts of " << Ticker << "!" << endl;

    }

};

class FixedIncomeConnection : public SecurityConnection
{

public:

    FixedIncomeConnection() {}
    ~FixedIncomeConnection() {}

    void buySecurity(string Ticker, uint numShares) override
    {

        cout << "You bought " << numShares << " loans of " << Ticker << "!" << endl;

    }

    void sellSecurity(string Ticker, uint numShares) override
    {

        cout << "You sold " << numShares << " loans of " << Ticker << "!" << endl;

    }

};

class ForexConnection : public SecurityConnection
{

public:

    ForexConnection() {}
    ~ForexConnection() {}

    void buySecurity(string Ticker, uint numShares) override
    {

        cout << "You exchanged for $" << numShares << " of " << Ticker << "!" << endl;

    }

    void sellSecurity(string Ticker, uint numShares) override
    {

        cout << "You exchanged out of $" << numShares << " of " << Ticker << "!" << endl;

    }

};

class Broker
{

public:

    string type;

    map<string, shared_ptr<SecurityConnection>> tickerConnections;

    shared_ptr<EquityConnection> stockConn;
    shared_ptr<DerivativeConnection> derivativeConn;
    shared_ptr<FixedIncomeConnection> fiConn;
    shared_ptr<ForexConnection> forexConn;

    Broker()
    {

        // Allocate the Connections to the Exchanges
        stockConn = make_shared<EquityConnection>();
        derivativeConn = make_shared<DerivativeConnection>();
        fiConn = make_shared<FixedIncomeConnection>();
        forexConn = make_shared<ForexConnection>();

        // Add example tickers for each Connection

        // Add Stock tickers
        tickerConnections["DPST"] = stockConn;
        tickerConnections["TQQQ"] = stockConn;
        tickerConnections["LABU"] = stockConn;
        tickerConnections["SOXL"] = stockConn;

        // Add Derivative tickers
        tickerConnections["DPST-CALL"] = derivativeConn;
        tickerConnections["DPST-PUT"] = derivativeConn;
        tickerConnections["TQQQ-CALL"] = derivativeConn;
        tickerConnections["TQQQ-PUT"] = derivativeConn;        
        tickerConnections["LABU-CALL"] = derivativeConn;
        tickerConnections["LABU-PUT"] = derivativeConn;        
        tickerConnections["SOXL-CALL"] = derivativeConn;
        tickerConnections["SOXL-PUT"] = derivativeConn;

        // Add Fixed Income tickers
        tickerConnections["912797FW2"] = fiConn;

        // Add Foreign Exchange tickers
        tickerConnections["USD-GBP"] = forexConn;
        tickerConnections["GBP-USD"] = forexConn;

    }
    ~Broker()
    {

        // Release the tickers
        tickerConnections.clear();

        // Release the Connections
        stockConn.reset();
        derivativeConn.reset();
        fiConn.reset();
        forexConn.reset();

    }

    void buySecurity(string Ticker, uint numShares)
    {

        tickerConnections[Ticker]->buySecurity(Ticker, numShares);

    }

    void sellSecurity(string Ticker, uint numShares)
    {

        tickerConnections[Ticker]->sellSecurity(Ticker, numShares);

    }

};

#endif // BROKER_H