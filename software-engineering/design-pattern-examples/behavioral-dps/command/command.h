/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef COMMAND_H
#define COMMAND_H

#include <iostream>
#include <string>

using namespace std;

class Command
{

public:

    virtual void run() {}

};

class AddCommand : public Command
{

public:

    AddCommand() {}

    void run() override
    {

        cout << "Now running git add" << endl;

    }

};

class RemoveCommand : public Command
{

public:

    void run() override
    {

        cout << "Now running git rm" << endl;

    }

};

class CommitCommand : public Command
{

public:

    void run() override
    {

        cout << "Now running git commit" << endl;

    }

};

class PushCommand : public Command
{

public:

    void run() override
    {

        cout << "Now running git push" << endl;

    }

};

#endif // COMMAND_H  
