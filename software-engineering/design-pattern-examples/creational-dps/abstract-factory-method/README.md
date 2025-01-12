## The Abstract Factory Method Pattern

The Abstract Factory Method Pattern is a Creational Design Pattern that defines an interface for creating a Factory that a system can use for creating objects.

The Abstract Factory Method Pattern can basically be viewed as an extension of the Factory Method pattern. The difference is that instead of directly using an abstract method for creating concrete classes, we use a concrete factory for doing so.

In this example, we have a program that allows a user to "allocate" furniture and appliances. The method that you would allocate those objects through would be by using these concrete factories that are represented as Stores.

Notice that main does not know what kind of stores we are allocating. All it knows is that we allocate stores and we also allocate furniture and appliances of which there are multiple kinds of each.