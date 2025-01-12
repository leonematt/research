/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include <vector>
#include <memory>

#include "command.h"

using namespace std;

int main(void)
{

    // Create list of Commands
    vector<shared_ptr<Command>> commandList;

    // Add Commands to list
    commandList.push_back(make_shared<AddCommand>());
    commandList.push_back(make_shared<CommitCommand>());
    commandList.push_back(make_shared<PushCommand>());

    // Execute each Command
    for (auto i : commandList) {

        i->run();
        i.reset();

    }

    // Reset the list of Commands
    commandList.clear();

    return 0;

}