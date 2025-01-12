/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

import java.util.*;

public class Orchestrator
{

    public int numberOfNodes;
    public int capQualities;
    private Vector<Node> nodes;

    public Orchestrator(int numNodes, int qualitiesChoice)
    {

        // Set up the orchestator with the nodes
        setupOrchestrator(numNodes, qualitiesChoice);

        System.out.println("Finished setting up the Orchestrator\n");

    }

    public void startOrchestrator()
    {

        switch(capQualities) {
            case 1:

                System.out.println("Running with Consistency and Availability\n");
                System.out.println("Sometimes the Network goes down\n");

                break;

            case 2:

                System.out.println("Running with Consistency and Partitioning\n");
                System.out.println("Sometimes the Node fails\n");

                break;
            case 3:

                System.out.println("Running with Availability and Partitioning\n");
                System.out.println("Sometimes the Node doesn't receive write requests\n");

                break;
        }

        System.out.println("Starting the Orchestrator\n");

        Random rng = new Random();

        int newValue;
        int actualValue;

        boolean continueProcessing = true;

        while (continueProcessing) {

            // Consistency and Availability
            if (capQualities == 1) {

                newValue = rng.nextInt(100);
                System.out.println("Writing " + newValue + " to the Nodes\n");
                for (int i = 0; i < numberOfNodes; i++)
                    nodes.get(i).write(newValue);

                newValue = rng.nextInt(numberOfNodes);

                System.out.println("Value stored in Node " + newValue + " is: " + nodes.get(newValue).read() + "\n");

                try {

                    Thread.sleep(500);

                } catch (InterruptedException e) {}

                if (rng.nextInt(10) <= 1) {

                    System.out.println("The Network Partition just failed and we are unable to recover\n");

                    continueProcessing = false;

                }

            }

            // Consistency and Partitioning
            if (capQualities == 2) {

                newValue = rng.nextInt(100);
                System.out.println("Writing " + newValue + " to the Nodes\n");
                for (int i = 0; i < numberOfNodes; i++)
                    nodes.get(i).write(newValue);


                    newValue = rng.nextInt(numberOfNodes);

                    if (rng.nextInt(10) <= 1) {

                        System.out.println("Node " + newValue + " that we want to read from is down\n");
    
                        continueProcessing = false;
    
                    }
                    else {

                        System.out.println("Value stored in Node " + newValue + " is: " + nodes.get(newValue).read() + "\n");

                    }

                try {

                    Thread.sleep(500);

                } catch (InterruptedException e) {}

            }

            // Availability and Partitioning
            if (capQualities == 3) {

                actualValue = rng.nextInt(100);
                System.out.println("Writing " + actualValue + " to the Nodes\n");
                for (int i = 0; i < numberOfNodes; i++)
                    if (rng.nextInt(100) >= 15)
                        nodes.get(i).write(actualValue);

                    newValue = rng.nextInt(numberOfNodes);

                    System.out.println("Value stored in Node " + newValue + " is: " + nodes.get(newValue).read() + "\n");
                    System.out.println("The actual value should be: " + actualValue + "\n");

                    if (nodes.get(newValue).read() != actualValue)

                        continueProcessing = false;

                try {

                    Thread.sleep(500);

                } catch (InterruptedException e) {}

            }

        }

    }

    private void setupOrchestrator(int numNodes, int qualities)
    {

        nodes = new Vector<Node>();

        for(int i = 0; i < numNodes; i++)
            nodes.add(new Node());

        System.out.println("We added " + numNodes + " nodes to the system\n\n");

        numberOfNodes = numNodes;
        capQualities = qualities;

    }

}