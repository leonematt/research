/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "store.h"

int main(void)
{

    // Appliance Store/Factory
    shared_ptr<ApplianceStore> AS = dynamic_pointer_cast<ApplianceStore>(getStore("Appliance"));

    // Furniture Store/Factory
    shared_ptr<FurnitureStore> FS = dynamic_pointer_cast<FurnitureStore>(getStore("Furniture"));

    // Get a Television from the Appliance Factory
    shared_ptr<Appliance> myTV = AS->getAppliance("Television");

    // Get a Couch from the Furniture Factory
    shared_ptr<Furniture> myCouch = FS->getFurniture("Couch");

    return 0;

}