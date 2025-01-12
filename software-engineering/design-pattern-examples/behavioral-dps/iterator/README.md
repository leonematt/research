## The Iterator Pattern

The Iterator Pattern is a Behavioral Design Pattern that creates an inteface for access a list of objects sequentially.

The general idea is that instead of needing the user to handle the underlying pointers and memory, we provide an interface
so that the user can easily handles an array of objects without needing to manually handle them.

This example we implemented the Iterator container object that works like the one in C++ does.
We pass an array of objects to the class, and then we can treat it as we would any other iterator in C++, 
such as an iterator for the Vector container.
