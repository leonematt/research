/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef RUNTIME_H
#define RUNTIME_H

#include <memory>

using namespace std;

// Concrete Runtime Class
class Runtime
{

public:

    static shared_ptr<Runtime> getRuntime()
    {

        if (!currentRuntime) {
            currentRuntime = shared_ptr<Runtime>(new Runtime);
        }

        return currentRuntime;

    }

    ~Runtime() {}

private:

    static shared_ptr<Runtime> currentRuntime;

    Runtime()  {}
    Runtime(const Runtime &copy) {}

};

shared_ptr<Runtime> Runtime::currentRuntime;

#endif // RUNTIME_H