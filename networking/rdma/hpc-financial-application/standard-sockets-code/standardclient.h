/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <string>
#include <string.h>

#include <fstream>
#include <iostream>

#include <memory>
#include <vector>

#include <chrono>

#include <filesystem>

#include <cerrno>

#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>

using namespace std;

#ifndef STANDARDCLIENT_H
#define STANDARDCLIENT_H

class StandardClient
{

public:

    StandardClient() {}
    ~StandardClient() {}

    void setupConnection(string, string);
    bool releaseConnection();

    int receiveData();
    int sendData(string);

    int getConnectionID();

private:

    int socketNum;

    sockaddr_in serverAddr;

    int clientSocket;

};

#endif // STANDARDCLIENT_H