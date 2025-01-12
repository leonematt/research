/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "rdmaserver.h"

static uint8_t recvBuffers[100][16];
static uint8_t sendBuffers[100][16];

struct rdma_addrinfo* clearAddrInfo(struct rdma_addrinfo* res)
{

    struct rdma_addrinfo *tmp = NULL;

    while(res) {
 
        tmp = res;
        res = res->ai_next;

        if (tmp->ai_src_addr) {
            free(tmp->ai_src_addr);
            tmp->ai_src_addr = NULL;
        }
        if (tmp->ai_dst_addr) {
            free(tmp->ai_dst_addr);
            tmp->ai_dst_addr = NULL;
        }
        if (tmp->ai_src_canonname) {
            free(tmp->ai_src_canonname);
            tmp->ai_src_canonname = NULL;
        }     
        if (tmp->ai_dst_canonname) {
            free(tmp->ai_dst_canonname);
            tmp->ai_dst_canonname = NULL;
        }
        if (tmp->ai_route) {
            free(tmp->ai_route);
            tmp->ai_route = NULL;
        }
        if (tmp->ai_connect) {
            free(tmp->ai_connect);
            tmp->ai_connect = NULL;
        }
        free(tmp);
        tmp = NULL;     
    }
    res = NULL;

    return res;

}

struct rdma_cm_id * RDMAServer::getConnectionID()
{

    return connectionID;

}


bool RDMAServer::setupConnection(string server, string portnum)
{

    struct rdma_addrinfo hints = {0};
    struct rdma_addrinfo* res  = NULL;
    struct ibv_qp_init_attr attr = {0}, initAttr = {0};
    struct ibv_qp_attr qpAttr = {};

    struct rdma_cm_id *tmpID;

    struct rdma_conn_param connectionParams = {0};

    //struct ibv_wc workCompletion;

	int ret = -1;

    // Set up the type of connection
    memset(&hints, 0, sizeof hints);
    hints.ai_flags      = RAI_PASSIVE;
    hints.ai_qp_type    = IBV_QPT_RC;
    hints.ai_port_space = RDMA_PS_TCP;

    // Get the server's addr info
    ret = rdma_getaddrinfo(server.c_str(), portnum.c_str(), &hints, &res);
    if (ret != 0) {
        cerr << "Failed to get addr info." << strerror(errno) << endl;
        goto BAD_ADDRINFO;
    }

    // Set the communication line details
    attr.cap.max_send_wr = 100;
    attr.cap.max_recv_wr = 100;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 16;
    attr.sq_sig_all = 1;

    // Create the communication endpoint
    ret = rdma_create_ep(&tmpID, res, NULL, &attr);
    if (ret != 0) {
        cerr << "Failed to create endpoint." << errno << endl;
        goto BAD_ENDPOINT;
    }

    // Set the endpoint to listen
    ret = rdma_listen(tmpID, 1);
    if (ret != 0) {
        cerr << "Server failed to listen." << endl;
        goto BAD_SERVER_CALL;
    }

    // Receive a request to connect
    ret = rdma_get_request(tmpID, &connectionID);
    if (ret != 0) {
        cerr << "Server failed to get request from listening." << endl;
        goto BAD_SERVER_CALL;
    }
    
    // Get the details of the queue pair from the incoming connection
    ret = ibv_query_qp(connectionID->qp, &qpAttr, IBV_QP_CAP, &initAttr);
    if (ret != 0) {
        cerr << "Server failed to query the QP." << endl;
        goto BAD_SERVER_CALL;
    }

    // Set up receive buffers
    for (int i = 0; i < 100; i++) {
        recvMRs.push_back(make_shared<struct ibv_mr *>(rdma_reg_msgs(connectionID, recvBuffers[i], 16)));
    }

    cout << "Populated receive memory regions" << endl;

    // Set up send buffer
    /*sendMR = rdma_reg_msgs(connectionID, sendBuffer, 16);
    if (sendMR == NULL) {
        ret = -1;
        cerr << "Server failed to register send buffer";
        goto BAD_SERVER_CALL;
    }*/
    for (int i = 0; i < 100; i++) {
        sendMRs.push_back(make_shared<struct ibv_mr *>(rdma_reg_msgs(connectionID, sendBuffers[i], 16)));
    }

    cout << "Populated send memory regions" << endl;

    // Prepare for incoming message
    ret = rdma_post_recv(connectionID, NULL, recvBuffers[0], 16, *recvMRs[0]);
    if (ret != 0) {
        cerr << "Server failed to receive connection message";
        goto BAD_SERVER_CALL;
    }               

    // Accept the connection
    ret = rdma_accept(connectionID, &connectionParams);
    if (ret != 0) {
        cerr << "Server failed to accept." << endl;
        goto BAD_SERVER_CALL;
    }

    /*while((ret = rdma_get_recv_comp(connectionID, &workCompletion)) == 0) {}

    ret = rdma_post_send(connectionID, NULL, sendBuffer, 16, sendMR, IBV_SEND_INLINE);
    if (ret != 0) {
        cerr << "Server failed to receive connection message";
        goto BAD_SERVER_CALL;
    }*/

    cout << "Server has connected to client" << endl << flush;

    goto SUCCESS;

BAD_SERVER_CALL:

    rdma_destroy_ep(tmpID);

BAD_ENDPOINT:
BAD_ADDRINFO:
SUCCESS:
    
    res = clearAddrInfo(res);

    return (ret == 0);
}

bool RDMAServer::releaseConnection(void)
{
    if (connectionID != NULL) {
        rdma_disconnect(connectionID);
        connectionID = NULL;
    }

    connectionIP = "";

   recvMRs.clear();
   sendMRs.clear();

    return true;

}

int RDMAServer::receiveData()
{

    int ret = 0;

    struct ibv_wc workCompletion;

    int numBytes = stoi((char*)(recvBuffers[0]));
    int numReqs = numBytes / 16 + 1;
        
    cout << "Number of bytes to receive: " << numBytes << endl;
    cout << "Number of reqs to post: " << numReqs << endl;

    ret = rdma_post_recv(connectionID, NULL, recvBuffers[0], 16, *recvMRs[0]);

    auto start = chrono::high_resolution_clock::now();

    // Post a send to start the transfer
    ret = rdma_post_send(connectionID, NULL, sendBuffers[0], 16, *sendMRs[0], IBV_SEND_INLINE);
    if (ret != 0) {
        cerr << "Failed to post send work request: " << errno << endl;
    }

    //We need to know the last sendBuffer that we put data in
    int i = 0;

    // Receive the data
    for (i = 0; i < 100 && numReqs > 0; ++i, --numReqs) {
        ret = rdma_post_recv(connectionID, NULL, recvBuffers[i], 16, *recvMRs[i]);
        while((ret = rdma_get_recv_comp(connectionID, &workCompletion)) == 0) {}
        ret = rdma_post_send(connectionID, NULL, sendBuffers[0], 16, *sendMRs[0], IBV_SEND_INLINE);

        if (i == 99) {
            i = -1;
        }

    }

    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    recvBuffers[i][0] = 0;

    // Dump the data
    cout << "FINAL DATA RECEIVED: " << endl;
    cout << (char *)recvBuffers[0];

    cout << endl << endl << "Time spent transmitting: " << duration.count() << " microseconds" << endl;

    cout << endl << endl << endl << "RDMA Reception Complete" << endl;

    return 0;

}

int RDMAServer::sendMSG(string msg)
{

    return 0;

}
