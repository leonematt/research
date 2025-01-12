## The Composite Pattern

The Composite Pattern is a Structural Design Pattern that composes objects into trees in part-whole hierarchies

The general idea is that by engineering our classes properly, we can treat a single object the same way that we
treat multiple objects.

In this example we have a bunch of hardware classes along with an overarching class called HardwareGroup. The HardwareGroup
contains a vector of hardware devices and you can treat it like its own hardware device.

By containing all of these like classes together in a single group, if we need to operate on all of the objects or a subset,
then we can do that.
