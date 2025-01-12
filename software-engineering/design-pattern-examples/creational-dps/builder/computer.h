/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef COMPUTER_H
#define COMPUTER_H

#include <string>

#include <iostream>

#include <vector>

#include <memory>

#include "CPU.h"
#include "MB.h"
#include "RAM.h"
#include "DISK.h"
#include "PS.h"

using namespace std;

class Computer
{

public:

    string description;

    //CPU *cCPU;
    //MB *cMB;
    //RAM *cRAM;
    //DISK *cDISK;
    //PS *cPS;

    shared_ptr<CPU> cCPU;
    shared_ptr<MB> cMB;
    shared_ptr<RAM> cRAM;
    shared_ptr<DISK> cDISK;
    shared_ptr<PS> cPS;

    Computer() {}
    ~Computer() {}

};

// Abstract Computer Builder Class
class ComputerBuilder
{

public:

    string computerBrand;

    virtual void buildComputer() {}

};

// Concrete Computer Builder Class
class IntelComputerBuilder : public ComputerBuilder
{

public:

    IntelComputerBuilder() {}
    ~IntelComputerBuilder() {}

    void buildComputer() override {

        //Computer *newComp = new Computer();
        shared_ptr<Computer> newComp = make_shared<Computer>();

        newComp->cCPU = getCPU();
        newComp->cMB = getMB();
        newComp->cRAM = getRAM();
        newComp->cPS = getPS();
        newComp->cDISK = getDISK();

        computersInStock.push_back(newComp);

    }

    shared_ptr<Computer> getComputer() 
    {

       // Computer *ret = computersInStock[0];
        shared_ptr<Computer> ret = computersInStock[0];

        computersInStock.pop_back();

        return ret;

    }

private:

    vector<shared_ptr<Computer>> computersInStock;

    shared_ptr<CPU> getCPU()
    {
        return make_shared<CPU>();
    }

    shared_ptr<MB> getMB()
    {
        return make_shared<MB>();
    }

    shared_ptr<RAM> getRAM()
    {
        return make_shared<RAM>();
    }

    shared_ptr<PS> getPS()
    {
        return make_shared<PS>();
    }

    shared_ptr<DISK> getDISK()
    {
        return make_shared<DISK>();
    }

};

// Director Computer Store Class
class ComputerStore
{

public:

    //IntelComputerBuilder *intelSupplier;
    shared_ptr<IntelComputerBuilder> intelSuppplier;

    ComputerStore() {

        intelSuppplier = make_shared<IntelComputerBuilder>();

    }

    ~ComputerStore() {}

    shared_ptr<Computer> getComputer(string computerBrand)
    {

        shared_ptr<Computer> ret;

        if (computerBrand.compare("Intel") == 0) {

            intelSuppplier->buildComputer();

            ret = intelSuppplier->getComputer();

        }

        return ret;

    }

};

#endif // COMPUTER_H