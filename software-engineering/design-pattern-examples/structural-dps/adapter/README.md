## The Adapter Pattern

The Adapter Pattern is a Structural Design Pattern that conforms interfaces from one system or library to another.

The general idea is that when we have a system that needs to talk to another system but the way it communicates is not developed within
the existing classes.

In this example, we have a Target, TargetAdapter, and Client class. The Client class needs to talk to the Target class. The Client
class sends a String to the Target, but the Target only accepts an integer. By writing the Target Adapter class and having the Client
talk to that in a way it understands, we can make the Client compatible with Target. This pattern is also useful for decoupling,
so we keep all of the interface-specific code away from the higher-level implementations and maintains separation of concerns.