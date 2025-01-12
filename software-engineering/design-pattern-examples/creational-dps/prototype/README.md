## The Prototype Pattern

The Prototype Pattern is a Creational Design Pattern that emphasizes object creation through cloning.

The general idea is that when you go to create a complex or sensitive object, it becomes advantageous to take an existing object.

In this example, we have a ServerConnection, which represents a connection to a service across a network or it could be a service running on the localhost like a database.

When we go to create a ServerConnection, we use a clone method, which passes us a pointer to a newly allocated ServerConnection.

Notice that in main we get a different address but the cloned object has the same value.

This pattern is also incredibly useful when you need to rapidly create similar objects in a simplified manner.