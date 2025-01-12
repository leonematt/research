/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

public class Node
{

    int currentValue;

    public Node() {}

    public void write(int newValue)
    {

        currentValue = newValue;

    }

    public int read()
    {

        return currentValue;

    }

}