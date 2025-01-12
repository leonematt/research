/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "connection.h"

int main(void)
{

    // Set up NetworkConnections
    shared_ptr<Connection> connectionGRPC = getConnection("Network", "GRPC");
    shared_ptr<Connection> connectionMPI  = getConnection("Network", "MPI");

    // Send messages using each Connection
    connectionGRPC->sendMessage("Hello World!");
    connectionMPI->sendMessage("Hello World!");

    return 0;

}