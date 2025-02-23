// test_runner.cc
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include "PipeCallbackHandler.hh"

using Clock = std::chrono::high_resolution_clock;
using namespace Drp;

// Dummy structure to simulate sc_DldEvent.
// Make sure this structure’s layout is compatible with what processScDldEvent() expects.
struct DummyDldEvent {
    uint64_t time_tag;  // Used as pulseid.
    uint16_t dif1;
    uint16_t dif2;
    uint64_t sum;
};

int main(int argc, char* argv[]) {
    // Set default options.
    int measurementTimeMs = 1000;
    std::string iniFilePath = "tdc_gpx3.ini";
    size_t batchSize = 100;

    // Command-line arguments (if provided) override defaults.
    // Expected arguments:
    //   argv[1] : measurementTimeMs (int)
    //   argv[2] : iniFilePath (string)
    //   argv[3] : batchSize (int)
    if (argc > 1)
        measurementTimeMs = std::stoi(argv[1]);
    if (argc > 2)
        iniFilePath = argv[2];
    if (argc > 3)
        batchSize = static_cast<size_t>(std::stoi(argv[3]));

    std::cout << "Using options:" << std::endl;
    std::cout << "  measurementTimeMs: " << measurementTimeMs << std::endl;
    std::cout << "  iniFilePath: " << iniFilePath << std::endl;
    std::cout << "  batchSize: " << batchSize << std::endl;

    // Create and initialize the handler.
    PipeCallbackHandler handler(measurementTimeMs, iniFilePath, batchSize);
    handler.init();
    handler.startMeasurement();

    // Use an atomic flag to signal when to stop consuming events.
    std::atomic<bool> stopFlag(false);
    std::atomic<size_t> eventCount(0);

    // Consumer thread: pop completed events as they become available.
    std::thread consumer([&]() {
        while (!stopFlag.load()) {
            KMicroscopeData event;
            if (handler.popEvent(event)) {
                eventCount++;
            } else {
                // Sleep briefly to avoid busy-waiting.
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    });

    // Let the test run for 5 seconds.
    constexpr double runTimeSec = 5.0;
    std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(runTimeSec)));
    stopFlag.store(true);
    consumer.join();

    double eventsPerSec = static_cast<double>(eventCount.load()) / runTimeSec;
    double kHz = eventsPerSec / 1000.0;
    std::cout << "Processed " << eventCount.load() << " events in "
              << runTimeSec << " seconds. Rate: " << kHz << " kHz" << std::endl;

    return 0;
}
