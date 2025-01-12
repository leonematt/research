## The Proxy Pattern

The Proxy Pattern is a Structural Design Pattern that creates a proxy access to a class.

The general idea is that once we have objects we might want to control accesses to those objects, which can
be viewed as the object-oriented equivalent to a software hook.

In this example we have Tunnel class that represents a Geneve or Ipsec tunnel. We also have a SessionHandler
class for creating the Tunnel. We have a proxy pattern in place so that the main program isn't directly creating tunnels, 
its just asking the SessionHandler to create the Tunnel.

By doing this, we can maintain a separation of concerns so that we don't just have one class doing all of the work.
That would be an anti-pattern.