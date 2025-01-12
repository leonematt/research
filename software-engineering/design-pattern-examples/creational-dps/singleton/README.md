## The Singleton Pattern

The Singleton Pattern is a Creational Design Pattern that emphasizes maintaining a single instance of a class.
It is also considered to be an anti-pattern depending on the context as well.

The general idea is that you want to maintain a single point of access to some resource, and the singleton is a way to do that. The singleton has a ton of application when it is used appropriately. For things like logging or caching, it can be a useful tool.

In this example, we have a Runtime class that is meant to pay homage to the Java Runtime Environment that frequently uses singletons in its implementation.

In Runtime, we set the Runtime's constructor and copy constructor as private methods, so you can't call it directly. The only method that you can call is the static getRuntime method, which you can call without having to use an instance of the class. This method allocates the Runtime if it has not been allocated yet and it returns itself at the end of the call.

Notice that in main we see that every time we get a copy of the Runtime, it points to the same virtual address, which is what we expect. This is basically like taking a data structure that you have already allocated and copying the pointer.

The reason why Singleton can be an anti-pattern is because if it is not implemented correctly, it becomes more of a hindrance than it is a benefit. A major problem is unit-testing where after each test you need to make sure it is reset to the minimal good state.