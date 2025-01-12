/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef FURNITURE_H
#define FURNITURE_H

#include <string>

using namespace std;

class Furniture 
{

public:

};

class Table : public Furniture {

public:

    Table() {}
    ~Table() {}

};

class Couch : public Furniture {

public:

    Couch() {}
    ~Couch() {}

};

class Chair : public Furniture {

public:

    Chair() {}
    ~Chair() {}

};

#endif //FURNITURE_H