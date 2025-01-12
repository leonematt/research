/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef CONNECTION_H
#define CONNECTION_H

#include <string>

#include <iostream>

#include <memory>

#include "message.h"

using namespace std;

// Abstract Interface
class Connection
{

public:

    virtual void sendMessage(string msg) {};

};

// Concrete Interface
class NetworkConnection : public Connection
{

public:

    NetworkConnection(shared_ptr<MessageInterface> interface) 
    {
        msgIntf = interface;
    }
    ~NetworkConnection() {}

    void sendMessage(string msg) override
    {
        msgIntf->sendMessage(msg);
    }

private:

    shared_ptr<MessageInterface> msgIntf;

};

// Concrete Interface
class DatabaseConnection : public Connection
{

public:

    DatabaseConnection() {}
    ~DatabaseConnection() {}

    void sendMessage(string msg) override
    {
        msgIntf->sendMessage(msg);
    }

private:

    shared_ptr<MessageInterface> msgIntf;

};

// Returns a Network Connection
shared_ptr<NetworkConnection> getNetworkConnection(string protocol)
{

    // The kind of network protocol
    shared_ptr<MessageInterface> msgIntf = NULL;
    
    if (protocol == "GRPC")
       msgIntf = make_shared<GRPCInterface>();
    
    else if (protocol == "MPI")
        msgIntf = make_shared<MPIInterface>();
    
    return make_shared<NetworkConnection>(msgIntf);

};


// Returns a Database Connection
shared_ptr<DatabaseConnection> getDatabaseConnection()
{

    return make_shared<DatabaseConnection>();

}

// Returns a new connection
shared_ptr<Connection> getConnection(string kind, string protocol)
{

    shared_ptr<Connection> ret;

    if (kind == "Network")
        ret = dynamic_pointer_cast<Connection>(getNetworkConnection(protocol));
    
    else if (kind == "Database")
        ret = dynamic_pointer_cast<Connection>(getDatabaseConnection());

    return ret;

}

#endif // CONNECTION_H