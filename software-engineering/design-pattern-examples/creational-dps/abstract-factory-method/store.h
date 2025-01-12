/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef STORE_H
#define STORE_H

#include <string>

#include <iostream>

#include <memory>

#include "appliance.h"
#include "furniture.h"

using namespace std;

// Abstract Store Class
class Store
{

public:

    string storeType;

    virtual ~Store() {}

private:

};

// Concrete Store Class for furniture
class FurnitureStore : public Store
{

public:

    FurnitureStore(string type)
    {
        storeType = type;
    }

    ~FurnitureStore() {}

    shared_ptr<Furniture> getFurniture(string type)
    {
      
        shared_ptr<Furniture> ret;

        if (type.compare("Table") == 0) {
            cout << "Getting a Table" << endl;
            ret = make_shared<Table>();
        }

        else if (type.compare("Couch") == 0) {
            cout << "Getting a Couch" << endl;
            ret = make_shared<Couch>();
        }

        else if (type.compare("Chair") == 0) {
            cout << "Getting a Chair" << endl;
            ret = make_shared<Chair>();
        }

        return ret;  

    }

};

// Concrete Store Class for appliances
class ApplianceStore : public Store
{

public:

    ApplianceStore(string type)
    {
        storeType = type;
    }
    ~ApplianceStore() {}

    shared_ptr<Appliance> getAppliance(string type)
    {

        shared_ptr<Appliance> ret;

        if (type.compare("Television") == 0) {
            cout << "Getting a Television" << endl;
            ret = make_shared<Television>();
        }

        else if (type.compare("Stove") == 0) {
            cout << "Getting a Stove" << endl;
            ret = make_shared<Stove>();
        }

        else if (type.compare("Refrigerator") == 0) {
            cout << "Getting a Refrigerator" << endl;
            ret = make_shared<Refrigerator>();
        }

        return ret;

    }

};


// Function for allocating a concrete Vehicle class
shared_ptr<Store> getStore(string typeOfStore)
{
    shared_ptr<Store> ret;

    if (typeOfStore.compare("Furniture"))
        ret = make_shared<FurnitureStore>(typeOfStore);

    else if (typeOfStore.compare("Appliance"))
        ret = make_shared<ApplianceStore>(typeOfStore);

    return ret;

}

#endif // STORE_H