/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef REQUEST_H
#define REQUEST_H

#include <iostream>
#include <string>
#include <memory>

using namespace std;

class Request
{

public:

    string rType;

    Request(string type)
    {

        rType = type;

    }
    ~Request() {}

};


class RequestHandler
{

public:

    shared_ptr<RequestHandler> nextHandler;

    virtual void handleRequest(shared_ptr<Request> request) {}

    void setNextHandler(shared_ptr<RequestHandler> handler)
    {

        nextHandler = handler;

    }

};

class AuthenticationHandler : public RequestHandler
{

public:

    void handleRequest(shared_ptr<Request> request) override
    {

        cout << "Authentication Handler handling Request" << endl;

        if (nextHandler)
            nextHandler->handleRequest(request);

    }

};

class ErrorHandler : public RequestHandler
{

public:

    void handleRequest(shared_ptr<Request> request) override
    {

        cout << "Error Handler handling Request" << endl;

        if (nextHandler)
            nextHandler->handleRequest(request);

    }

};

class LogHandler : public RequestHandler
{

public:

    void handleRequest(shared_ptr<Request> request) override
    {

        cout << "Logging Handler handling request" << endl;

        if (nextHandler)
            nextHandler->handleRequest(request);

    }

};

#endif // REQUEST_H  
