/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "session.h"

int main(void)
{

    // Create a SessionHandler
    shared_ptr<SessionHandler> handler = make_shared<SessionHandler>();

    // Create a Tunnel
    shared_ptr<Tunnel> exampleTunnel = handler->createTunnel();

    // Delete the Tunnel
    exampleTunnel.reset();

    return 0;

}