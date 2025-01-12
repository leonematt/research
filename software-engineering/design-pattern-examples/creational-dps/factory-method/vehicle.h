/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef VEHICLE_H
#define VEHICLE_H

#include <string>

#include <iostream>

#include <memory>

using namespace std;

// Abstract Vehicle Class
class Vehicle
{

public:

    string vehicleType;

private:

};

// Concrete Vehicle Class for short distances
class Bike : public Vehicle
{

public:

    Bike(string name)
    {
        vehicleType = name;
    }

    ~Bike() {}

};

// Concrete Vehicle Class for medium to long distances
class Car : public Vehicle
{

public:

    Car(string name)
    {
        vehicleType = name;
    }
    ~Car() {}

};

// Concrete Vehicle Class for long distances
class Plane : public Vehicle
{

public:

    Plane(string name)
    {
        vehicleType = name;
    }
    ~Plane() {}

};

// Function for allocating a concrete Vehicle class
shared_ptr<Vehicle> getVehicle(uint distanceInMiles)
{
    shared_ptr<Vehicle> ret = NULL;

    if (distanceInMiles <= 3.0) {
        cout << "Because your distance is less than 3 miles, we gave you a bike" << endl;
        ret = make_shared<Bike>("Bike");
    }

    else if (distanceInMiles > 3.0 && distanceInMiles < 300.0) {
        cout << "Because your distance is greater than 3 miles and less than 300 miles, we gave you a car" << endl;
        ret = make_shared<Car>("Car");
    }

    else {
        cout << "Because your distance is greater than 300 miles, we gave you a plane" << endl;
        ret = make_shared<Plane>("Plane");
    }

    return ret;

}

#endif // VEHICLE_H