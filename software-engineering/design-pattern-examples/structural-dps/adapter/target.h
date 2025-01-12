/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef TARGET_H
#define TARGET_H



class Target
{

public:

    int targetValue;

    Target() {}
    ~Target() {}

    void setTargetValue(int value)
    {
        targetValue = value;
    }

    int getData()
    {

        return targetValue;

    }

};

#endif // TARGET_H