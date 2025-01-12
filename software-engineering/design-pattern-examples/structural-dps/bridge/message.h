/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>

#include <iostream>

using namespace std;

// Abstract Interface
class MessageInterface
{

public:

    virtual void sendMessage(string msg) {};

};

// Concrete Interface
class GRPCInterface : public MessageInterface
{

public:

    GRPCInterface() {}
    ~GRPCInterface() {}

    void sendMessage(string msg) override
    {
        cout << endl << "Sending GRPC Message: " << msg << endl;
    }

};

// Concrete Interface
class MPIInterface : public MessageInterface
{

public:

    MPIInterface() {}
    ~MPIInterface() {}

    void sendMessage(string msg) override
    {
        cout << endl << "Sending MPI Message: " << msg << endl;
    }

};

#endif // MESSAGE_H