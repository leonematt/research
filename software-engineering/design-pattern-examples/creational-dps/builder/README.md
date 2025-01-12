## The Builder Pattern

The Builder Pattern is a Creational Design Pattern that gives a well-engineered manner for creating a complex object.

In this example, we have a program that allows a user to "build" a computer. The ComputerBuilder class is an abstract class and we have an IntelComputerBuilder as a concrete class that allocates and sets up the computer. The computer has some basic components that get set up by its constructor which is the beauty of the Builder PatternThe method that you would allocate those objects through would be by using these concrete factories that are represented as Stores.

Notice that main does not know what kind of computer we are allocating, just that it is allocating a computer. This example actually pairs the Abstract Method Pattern with the Builder Pattern.