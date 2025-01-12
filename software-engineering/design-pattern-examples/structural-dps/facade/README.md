## The Facade Pattern

The Facade Pattern is a Structural Design Pattern that gives a simple interface to a series of complex systems.

The general idea is that instead of having to duplicate a lot of code, we can refactor it into a much simpler interface.

In this example we have a bunch exchange connections designed to simulate the financial system. By placing all of these systems in a single class, we can make it really easy to use any of the objects with a small code and not much performance impact. We have a bunch of exchange connections classes like Stock, Derivatives, Fixed Income, and Foreign Currency Exchange classes, but we can just use the single, simple Broker class to interface with each of these complex systems.
