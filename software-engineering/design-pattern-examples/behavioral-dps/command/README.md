## The Command Pattern

The Command Pattern is a Behavioral Design Pattern that emphasizes handling of objects.

The general idea is that you wrap requests into their own objects. By wrapping a request into an object, we
can decouple requests from parts of our software stack that we want to keep separated in terms of concerns.

In this example we have a series of Command objects to represent Git commands. By wrapping the Git Add, Rm, Commit, and Push
commands as their own objects. When we use Git, we pass it commands that we don't necessarily know how they work, they do
what the documentation says it does. This decouples us from the low-level details of the system, so we just need to know what
commands we need to run.
