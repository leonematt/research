## The Decorator Pattern

The Decorator Pattern is a Structural Design Pattern that applies layers to existing objects.

The general idea is that by when we want classes to enhance their functionality, it may be beneficial
to apply layers or decorations to an object.

In this example we have Streams of data. If we want to do something like encrypt the data or compress it, instead of having a CompressedStream, EncryptedStream, and then the combinations of those streams,
we just add a Decorator class that we can dynamically apply more layers to, so we can transform a
Stream into an Encrypted or a Compressed Stream.

By layering on decorations instead of adding more and more classes, we keep our class hierarchies
much simpler by avoiding having to add more and more specialized classes.
