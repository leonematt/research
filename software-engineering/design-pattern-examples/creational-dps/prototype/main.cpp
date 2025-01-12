/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

# include "server.h"

#include <iostream>

using namespace std;

int main(void)
{

    // Hardcoded socket descriptor
    int socketVal = 10;

    // Connection 1
    shared_ptr<Connection> serverConn1 = make_shared<ServerConnection>(socketVal);

    // Connection 2
    shared_ptr<Connection> serverConn2 = serverConn1->clone();

    // Dump the class data
    cout << endl;
    cout << "Connection 1 Address: " << serverConn1 << endl;
    cout << "Connection 2 Address: " << serverConn2 << endl;
    cout << endl;
    cout << "Connection 1 Descriptor: " << serverConn1->socketVal << endl;
    cout << "Connection 2 Descriptor: " << serverConn2->socketVal << endl;

    return 0;

}