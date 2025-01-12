## The State Pattern

The State Pattern is a Behavioral Design Pattern that allows an object to change states
to change to new functionality.

The general idea is that when we have an object that can be one of many objects and should
be able to behave differently based on a context, we should be able to change the State
easily. This allows us to have more maintainable code.

In this example, we model a couple of trading strategies. We have a High Frequency Trading(HFT)
strategy, a Medium Frequency Trading(MFT) strategy, and a Low Frequency Trading(LFT) strategy.
We have a TradingContext class with a trading strategy member that we can rapidly change from
one strategy type to another easily.