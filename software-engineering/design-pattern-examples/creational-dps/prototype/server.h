/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef SERVER_H
#define SERVER_H

#include <memory>

using namespace std;

// Abstract Connection Class
class Connection
{

public:

    virtual shared_ptr<Connection> clone() const = 0;

    int socketVal;

};

// Concrete Connection Clss
class ServerConnection : public Connection
{

public:

    ServerConnection(int inputSocket)
    {

        socketVal = inputSocket;

    }
    ~ServerConnection() {}

    shared_ptr<Connection> clone() const override
    {
        return make_shared<ServerConnection>(socketVal);
    }

};

#endif // SERVER_H