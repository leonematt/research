## The Observer Pattern

The Observer Pattern is a Behavioral Design Pattern that creates a one-to-many relationship
between an object and a bunch of other objects.

The general idea is that if we want to implement an event-driven architecture, where an
event propagates to other components in the software system. In this kind of system we would
have an object that receives an event, and sends that event to a collection of objects that it
has access to.

This example implements 2 different classes, a Kernel and DeviceDriver class. The Kernel has
a collection of registered DeviceDrivers that the Kernel can instruct to behave in a certain way.
In this case, if we want the Kernel to switch to a Low Power Mode, we tell the Kernel, and the Kernel
tells all of the other subsystems to enter Low Power Mode, including its registered DeviceDrivers.
