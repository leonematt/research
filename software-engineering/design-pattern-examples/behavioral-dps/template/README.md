## The Template Pattern

The Template Pattern is a behavioral design pattern designed to force all subclasses
to run a specific functionality.

The general idea is that if you want all subclasses to behave the same way from a superclass,
you can create a template method that cannot be overridden or overloaded that will behave the same
for each subclass. Its important to note that not all of the functionality has to be the same for
each subclass

In this example, we have several Trader classes. We have an abstract SecuritiesTrader class that
will be used by each of its subclasses, StockTrader and OptionsTrader. The Template pattern comes into
play with the trade method. The method contains a placeOrder and logTrade method. The placeOrder method can be overridden, but the logTrade method is going to be the same across all of the subclasses, so when we trade a stock or an option, the logging functionality is the same for each Trader.