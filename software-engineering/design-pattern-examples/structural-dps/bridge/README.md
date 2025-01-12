## The Bridge Pattern

The Bridge Pattern is a Structural Design Pattern that makes abstract interfaces instead of only concrete interfaces.

The general idea is that we want to separate abstractions from implementations, so they are not tightly-coupled.

In this example, we have a Connection class that supports Network and Database connections. The Network Connection
supports GRPC and MPI message-passing protocols. The important thing to know is that the Messages are implemented
as abstractions, and the protocol messaging interfaces are concrete classes, including GRPC and MPI.

By passing a concrete messaging interface to the Connection class, we can make sure that our concrete classes are
not tightly coupled to their abstractions.