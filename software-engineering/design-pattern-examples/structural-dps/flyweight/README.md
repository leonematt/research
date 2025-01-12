## The Flyweight Pattern

The Flyweight Pattern is a Structural Design Pattern that allows for efficient creation of compelex objects.

The general idea is that if we have some commonality between the objects, we can just share references to the data members that are going to be the same
across all of the objects.

In this example we have a Function object made to represent an AWS Lamda Function and a Context object that we can share across all of the Functions being created.
By reusing the Context across all of the Functions, we minimize time and space that would be spent allocating new Contexts for each Function. By sharing a reference
with each Function, we minimize time and space of the Function creation.