/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "standardserver.h"

void StandardServer::setupConnection(string addr, string portNum)
{

    // Create connection socket
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);

    // Set up socket address
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(atoi(portNum.c_str()));
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    // Store address in serverAddr
    inet_pton(AF_INET, addr.c_str(), &(serverAddr.sin_addr));

    // Bind the socket
    bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr));

    // Listen on the socket
    listen(serverSocket, SOMAXCONN);

    // Get connection
    sockaddr_in clientAddr;
    socklen_t clientSize = sizeof(clientAddr);
    serverSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientSize);

    cout << "Client connected" << endl;

}

int StandardServer::receiveData()
{

    char dataBuffer[16] = {0};

    // Get first set of bytes
    send(serverSocket, dataBuffer, 16, 0);

    recv(serverSocket, dataBuffer, 16, 0);

    int numBytes = stoi(dataBuffer);
    int numReqs = numBytes / 16 + 1;

    auto start = chrono::high_resolution_clock::now();

    for (; numReqs > 0; numReqs--)
        recv(serverSocket, dataBuffer, 16, 0);

    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << endl << endl << "Time spent transmitting: " << duration.count() << " microseconds" << endl;

    cout << endl << endl << endl << "RDMA Reception Complete" << endl;

    return 0;

}