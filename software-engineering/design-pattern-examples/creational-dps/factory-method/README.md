## The Factory Method Pattern

The Factory Method Pattern is a Creational Design Pattern that defines an interface for creating objects.

The interface that is used may be a defined interface that you might find in Java, but it could also be defining an object
using its base class in C++.

In this example, we have an application that is written in main.cpp. We also have a library of classes implemented in vehicle.h.

In main.cpp, you will see that it only knows that vehicle.h is a library that allows it to allocate a vehicle.
It is important that it knows about the abstract Vehicle class, but that is it. It does not need to know about the concrete classes.

In vehicle.h, you will see that it implements a base vehicle class, a bike, car, and plane class, and a function
for deciding which class it is going to allocate. 

Notice that main doesn't exactly know what classes are implemented in vehicle.h. It just knows that it can use a function for getting
a vehicle, and it needs to pass the miles that need to be travelled to that function and the function will give it a vehicle that is
appropriate for the amount of distance.

The beauty of this pattern is that if the library changes by adding a new concrete class like a new vehicle, the code in main does not 
need to change at all. In this example, the code in main is coupled enough with the vehicle library so that it can use the library and 
it can function without it being so dependent that it needs to change when the library changes. This makes the code flexible.
