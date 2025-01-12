## The Chain-Of-Responsibility Pattern

The Chain-Of-Responsibility Pattern is a Behavioral Design Pattern that emphasizes handling of objects.

The general idea is that when you have objects that might have multiple handlers or systems for handling objects,
it becomes a good idea to consider building a sequence of handlers when you don't necessarily know at compile-time
what kind of handling system you want to use for dynamically created objects.

In this example we have a Request object that we might see in a middleware software. We have multiple handlers
that get called by different kinds of Request objects. We have multiple kinds of handlers, including
Authentication, Error, and Logging handlers, and they all contain a pointer to a handler that will be called in succession.
Because we decoupled the main program from the handlers, we can mix-and-match mutliple handlers together as we see fit instead
of maintaining a collection of pointers. In the main program we generate mutliple handlers and multiple requests and we can
see the chain come into action.
