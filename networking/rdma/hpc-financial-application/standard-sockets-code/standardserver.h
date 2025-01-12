/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <string>

#include <iostream>

#include <chrono>

#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>

using namespace std;

#ifndef STANDARDSERVER_H
#define STANDARDSERVER_H

class StandardServer
{

public:

    StandardServer() {}
    ~StandardServer() {}

    void setupConnection(string, string);
    bool releaseConnection();

    int receiveData();
    int sendMSG(string msg);

    int getConnectionID();

private:

    int socketNum;

    sockaddr_in serverAddr;

    int serverSocket;
    
};

#endif // STANDARDSERVER_H