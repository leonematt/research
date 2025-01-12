/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "rdmaclient.h"

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

struct rdma_cm_id * RDMAClient::getConnectionID()
{

    return connectionID;

}

bool RDMAClient::setupConnection(string server, string portnum)
{

    struct rdma_addrinfo hints = {0};
    struct rdma_addrinfo* res  = NULL;
    struct ibv_qp_init_attr attr = {0};
    struct rdma_cm_id *tmpID;

    struct rdma_conn_param connectionParams = {0};

	int ret = -1;

    // Set up the type of connection
    hints.ai_port_space = RDMA_PS_TCP;

    // Get the server's addr info
    ret = rdma_getaddrinfo(server.c_str(), portnum.c_str(), &hints, &res);
    if (ret != 0) {
        cerr << "Failed to get addr info." << ret << endl;
        goto BAD_ADDRINFO;
    }

    // Set the communication line details
    memset(&attr, 0, sizeof attr);
    attr.cap.max_send_wr = 100;
	attr.cap.max_recv_wr = 100;
	attr.cap.max_send_sge = 1;
	attr.cap.max_recv_sge = 1;
	//attr.cap.max_inline_data = 16;
    attr.sq_sig_all = 1;
    attr.qp_context = tmpID;

    // Create the communication endpoint
    ret = rdma_create_ep(&tmpID, res, NULL, &attr);
    if (ret != 0) {
        cerr << "Failed to create endpoint." << strerror(errno) << endl;
        goto BAD_ENDPOINT;
    }
    
    connectionID = tmpID;

    // Set up receive buffers
    for (int i = 0; i < 100; i++) {
        recvMRs.push_back(make_shared<struct ibv_mr *>(rdma_reg_msgs(connectionID, recvBuffers[i], 16)));
    }

    cout << "Populated receive memory regions" << endl;

    for (int i = 0; i < 100; i++) {
        sendMRs.push_back(make_shared<struct ibv_mr *>(rdma_reg_msgs(connectionID, sendBuffers[i], 16)));
    }

    cout << "Populated send memory regions" << endl;

    // Receive connection request response
    ret = rdma_post_recv(connectionID, NULL, recvBuffers[0], 16, *recvMRs[0]);
    if (ret) {
        cerr << "Client failed to post receive work request." << endl;
        goto BAD_CLIENT_CALL;
    }

    // Connect to server
    ret = rdma_connect(connectionID, &connectionParams);
    if (ret != 0) {
        cerr << "Client failed to connect." << endl;
        goto BAD_CLIENT_CALL;
    }

    cout << "Client has connected to the server" << endl << flush;

    goto SUCCESS;

BAD_CLIENT_CALL:

    rdma_destroy_ep(tmpID);

BAD_ENDPOINT:
BAD_ADDRINFO:

SUCCESS:
    
    res = clearAddrInfo(res);

    return (ret == 0);
}

bool RDMAClient::releaseConnection()
{
    if (connectionID != NULL) {
        rdma_disconnect(connectionID);
        connectionID = NULL;
    }

    connectionIP = "";

    /*if (recvMR != NULL) {
       ibv_dereg_mr(recvMR);
        recvMR = NULL;
    }

    if (sendMR != NULL) {
       ibv_dereg_mr(sendMR);
        sendMR = NULL;
    }*/

    recvMRs.clear();
    sendMRs.clear();
    
    return true;

}

int RDMAClient::receiveMSG(string& msg)
{

    return 0;

}

int RDMAClient::sendData(string fname)
{

    struct ibv_wc workCompletion;

    string dataLine;

    ifstream dataFile(fname, ios::binary | ios::in);
    if (!dataFile) {
        cerr << "Failed to open file: " << strerror(errno) << endl;
    }

    dataFile.seekg(0, dataFile.end);

    int fsize = (int)dataFile.tellg();

    dataFile.seekg(0, dataFile.beg);

    int numReqs = fsize / 16 + 1;

    memcpy(sendBuffers[0], to_string(fsize).c_str(), 16);

    // Send the number of bytes that will be sent over
    int ret = rdma_post_send(connectionID, NULL, sendBuffers[0], 16, *sendMRs[0], IBV_SEND_INLINE);
    if (ret != 0) {
        cerr << "Failed to post send work request: " << errno << endl;
    }

    // Wait for acknowledgement
    while((ret = rdma_get_recv_comp(connectionID, &workCompletion)) == 0) {}

    //We need to know the last sendBuffer that we put data in
    int i = 0;

    for (i = 0; i < 100 && numReqs > 0; ++i, --numReqs) {
        dataFile.read((char*)&sendBuffers[i][0], 16);
        ret = rdma_post_recv(connectionID, NULL, recvBuffers[i], 16, *recvMRs[i]);
        rdma_post_send(connectionID, NULL, sendBuffers[i], 16, *sendMRs[i], IBV_SEND_INLINE);

        while((ret = rdma_get_recv_comp(connectionID, &workCompletion)) == 0) {}

        if (i == 99) {
            i = -1;
        }

    }

    sendBuffers[i][0] = 0;

    // Dump the data
    cout << "FINAL DATA SENT: " << endl;
    cout << (char *)sendBuffers[0];

    cout << endl << endl << endl << "RDMA Transfer Complete" << endl;

    return 0;

}