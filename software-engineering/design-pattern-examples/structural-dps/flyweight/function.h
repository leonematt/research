/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef FUNCTION_H
#define FUNCTION_H

#include <memory>

#include "context.h"

using namespace std;

class Function
{

public:

        Function() {

            currentContext = make_shared<Context>();

        }
        ~Function() {}

        shared_ptr<Function> cloneFunction() {

            shared_ptr<Function> ret = make_shared<Function>();

            ret->currentContext = this->currentContext;

            return ret;

        }

        shared_ptr<Context> getContext() {

            return currentContext;

        }

private:
    
    int functionNum;

    shared_ptr<Context> currentContext;

};

#endif // FUNCTION_H
