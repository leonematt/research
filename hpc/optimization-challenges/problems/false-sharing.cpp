/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/


#include <iostream>
#include <thread>
#include <vector>

constexpr int NUM_ITERATIONS = 1000000000;
constexpr int NUM_THREADS = 2;

// Two integers placed side by side, likely to be in the same cache line
struct Data {
    long a;
    long b;
} data;

void increment_a() {
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        data.a++;
    }
}

void increment_b() {
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        data.b++;
    }
}

int main() {

    // Set up threads
    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();
    // BEGIN HOTPATH CODE

    threads.emplace_back(increment_a);
    threads.emplace_back(increment_b);

    for (auto& t : threads) {
        t.join();
    }

    // END HOTPATH CODE
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Execution time: " << duration << " ms\n";

    return 0;
}
