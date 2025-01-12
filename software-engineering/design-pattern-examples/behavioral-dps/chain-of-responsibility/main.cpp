/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "request.h"

int main(void)
{

    // Create the Logging Handler
    shared_ptr<RequestHandler> lHandler = make_shared<LogHandler>();

    // Create the Authentication Handler
    shared_ptr<RequestHandler> aHandler = make_shared<AuthenticationHandler>();
    aHandler->setNextHandler(lHandler);

    // Create the Error Handler
    shared_ptr<RequestHandler> eHandler = make_shared<ErrorHandler>();
    eHandler->setNextHandler(lHandler);

    // Create an AuthenticationRequest
    shared_ptr<Request> aReq = make_shared<Request>("Authentication Request");

    // Create an ErrorRequest
    shared_ptr<Request> eReq = make_shared<Request>("Error Request");

    // Handle both requests
    aHandler->handleRequest(aReq);
    eHandler->handleRequest(eReq);

    // Free the Requests
    aReq.reset();
    eReq.reset();

    return 0;

}