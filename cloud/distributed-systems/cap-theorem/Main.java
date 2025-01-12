/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

import java.io.*;
import java.util.*;

public class Main
{

    public static Orchestrator o;

    public static void main(String args[])
    {

        Scanner inputScanner = new Scanner(System.in);

        // Get the input from the leader
        boolean validInput = false;
        int choice = -1;
        while (validInput == false) {
            printHeader();
            choice = Integer.parseInt(inputScanner.nextLine());
            if (choice > 0 && choice < 4)
                validInput = true;
            else
                System.out.println("\n\nYou gave an invalid input, give a number between 1 and 3\n\n");
        }

        System.out.println("\n");

        o = new Orchestrator(10, choice);

        o.startOrchestrator();

    }

    public static void printHeader()
    {

        System.out.println("Give the integer that corresponds with the CAP combination that you want to model");
        System.out.println("1: CA - Consistency  and Availability");
        System.out.println("2: CP - Consistency  and Partitioning");
        System.out.println("3: AP - Availability and Partitioning");

    }

}