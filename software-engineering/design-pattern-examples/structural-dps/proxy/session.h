/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef SESSION_H
#define SESSION_H

#include <string>
#include <memory>
#include <iostream>
#include <random>

using namespace std;

class Tunnel
{

public:

    int authenticationKey;

    Tunnel(int tunnelKey)
    {
    
        authenticationKey = tunnelKey;
        
        cout << "Created a Tunnel" << endl;        
    
    }
    ~Tunnel() {}

};

class SessionHandler
{

public:

    SessionHandler() {}
    ~SessionHandler() {}
    
    shared_ptr<Tunnel> createTunnel()
    {
    
        random_device rngd;
        
        return make_shared<Tunnel>(rngd());
    
    }

};

#endif // SESSION_H
