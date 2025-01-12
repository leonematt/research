#include <iostream>
#include <thread>
#include <vector>

constexpr int NUM_ITERATIONS = 1000000000;
constexpr int NUM_THREADS = 2;

struct Data {
    long a;
    volatile char padding[64];
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

/*

   The solution for this challenge is to add padding between the shared datastruct
   members so that they don't share the same cacheline, thus the threads are not
   constantly invalidating each other's caches.

*/

int main() {
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
