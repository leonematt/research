/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef APPLIANCE_H
#define APPLIANCE_H

#include <string>

using namespace std;

class Appliance 
{

public:

};

class Television : public Appliance {

public:

    Television() {}
    ~Television() {}

};

class Stove : public Appliance {

public:

    Stove() {}
    ~Stove() {}

};

class Refrigerator : public Appliance {

public:

    Refrigerator() {}
    ~Refrigerator() {}

};

#endif //APPLIANCE_H