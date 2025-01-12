## The Mediator Pattern

The Mediator Pattern is a Behavioral Design Pattern that loosely couples the communication between objects

The general idea is that instead of writing code so that objects can communicate with each other,
write a mediator object that can handle the necessary communications between two objects. The benefit
is that we can minimize the coupling between objects at the expense of coupling them to the mediator
class.

This example implements several different engines in a trading system. We have a trading engine
which determines whether or not we "buy" or sell some stock. In this case we just tell it to buy
stock. We have an order engine that will "place the orders" on an exchange. We also have a
TSMediator class that handles the communication between the engines so that the engines are not
coupled to each other.