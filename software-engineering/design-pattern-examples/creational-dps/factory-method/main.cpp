/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

# include "vehicle.h"

int main(void)
{

    // Vehicle for short distances
    shared_ptr<Vehicle> VSD = getVehicle(1.0);

    // Vehicle for medium distances
    shared_ptr<Vehicle> VMD = getVehicle(5.0);

    // Vehicle for long distances
    shared_ptr<Vehicle> VLD = getVehicle(1000.0);

    return 0;

}