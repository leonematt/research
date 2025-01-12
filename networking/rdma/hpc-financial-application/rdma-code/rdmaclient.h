/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <string>

#include <fstream>
#include <iostream>

#include <memory>
#include <vector>

#include <netdb.h>

#include <rdma/rdma_verbs.h>
#include <rdma/rdma_cma.h>

using namespace std;

#ifndef RDMACLIENT_H
#define RDMACLIENT_H

class RDMAClient
{

public:

    string devName;

    // The memory regions for data communications
    vector<shared_ptr<struct ibv_mr *>> recvMRs;
    vector<shared_ptr<struct ibv_mr *>> sendMRs;

    RDMAClient(string deviceName)
    {

        devName = deviceName;

    }
    ~RDMAClient() 
    {

        releaseConnection();

    }

    bool setupConnection(string, string);
    bool releaseConnection();

    struct rdma_cm_id *getConnectionID();

    int receiveMSG(string& msg);
    int sendData(string fname);


private:

    // The communication socket
    struct rdma_cm_id *connectionID = NULL;

    // The IP that we connected to or binded to
    string connectionIP;

};

#endif // RDMACLIENT_H