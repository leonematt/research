/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "standardclient.h"

void StandardClient::setupConnection(string addr, string portNum)
{

    // Create connection socket
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    // Set up socket address
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(atoi(portNum.c_str()));

    // Store address in serverAddr
    inet_pton(AF_INET, addr.c_str(), &(serverAddr.sin_addr));

    // Connect to the server
    connect(clientSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr));

    cout << "Connected to the server" << endl;

}

int StandardClient::receiveData()
{

    char dataBuffer[16] = {};

    // Get first set of bytes
    recv(clientSocket, dataBuffer, 16, 0);

    int numBytes = stoi(dataBuffer);
    int numReqs = numBytes / 16 + 1;

    send(clientSocket, "OKAY", 4, 0);

    auto start = chrono::high_resolution_clock::now();

    for (; numReqs > 0; numReqs--)
        recv(clientSocket, dataBuffer, 16, 0);

    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << endl << endl << "Time spent transmitting: " << duration.count() << " microseconds" << endl;

    cout << endl << endl << endl << "RDMA Reception Complete" << endl;

    return 0;

}

int StandardClient::sendData(string fname)
{
    char dataBuffer[16] = {0};

    int numBytes;
    int numReqs;

    ifstream dataFile(fname, ios::binary | ios::in);
    if (!dataFile) {
        cerr << "Failed to open file: " << fname << endl;
    }

    dataFile.seekg(0, dataFile.end);

    int fsize = (int)dataFile.tellg();

    dataFile.seekg(0, dataFile.beg);

    numBytes = fsize;
    numReqs = numBytes / 16 + 1;

    memcpy(dataBuffer, to_string(numBytes).c_str(), sizeof(numBytes));

    cout << "Sending: " << numBytes << endl;

    send(clientSocket, dataBuffer, 16, 0);

    recv(clientSocket, dataBuffer, 16, 0);

    for (; numReqs > 0; numReqs--)
        send(clientSocket, dataBuffer, 16, 0);

    return 0;

}