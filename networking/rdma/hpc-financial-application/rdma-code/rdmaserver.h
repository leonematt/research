/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <cerrno>

#include <chrono>

#include <string>

#include <iostream>

#include <memory>
#include <vector>

#include <netdb.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_verbs.h>
#include <rdma/rdma_cma.h>

using namespace std;

#ifndef RDMASERVER_H
#define RDMASERVER_H

class RDMAServer
{

public:

    string devName;

    // The memory regions for data communications
    vector<shared_ptr<struct ibv_mr *>> recvMRs;
    vector<shared_ptr<struct ibv_mr *>> sendMRs;

    RDMAServer(string deviceName)
    {

        devName = deviceName;

    }
    ~RDMAServer()
    {

        releaseConnection();

    }

    bool setupConnection(string, string);
    bool releaseConnection();

    struct rdma_cm_id *getConnectionID();

    int receiveData();
    int sendMSG(string msg);


private:

    // The communication socket
    struct rdma_cm_id *connectionID = NULL;

    // The IP that we connected to or binded to
    string connectionIP;

};



#endif // RDMASERVER_H