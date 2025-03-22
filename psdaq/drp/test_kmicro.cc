#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include "PipeCallbackHandler.hh"

using Clock = std::chrono::high_resolution_clock;
using namespace Drp;

int main(int argc, char* argv[]) {
    // Set default options.
    int measurementTimeMs = 1000;
    size_t queueCapacity = 65536;

    // Command-line arguments (if provided) override defaults.
    if (argc > 1) measurementTimeMs = std::stoi(argv[1]);
    std::string iniFilePath = (argc > 2 && argv[2] != nullptr) ? std::string(argv[2]) : "tdc_gpx3.ini";
    if (argc > 3) queueCapacity = static_cast<size_t>(std::stoi(argv[3]));

    std::cout << "Using options:" << std::endl;
    std::cout << "  measurementTimeMs: " << measurementTimeMs << std::endl;
    std::cout << "  iniFilePath: " << iniFilePath << std::endl;
    std::cout << "  queueCapacity: " << queueCapacity << std::endl;

    // Create and initialize the handler.
    PipeCallbackHandler handler(measurementTimeMs, iniFilePath, queueCapacity);
    handler.init();
    handler.startMeasurement();

    // Use atomic flags to track event stats.
    std::atomic<bool> stopFlag(false);
    std::atomic<size_t> eventCount(0);
    std::atomic<size_t> intervalEventCount(0);
    std::atomic<size_t> corruptEventCount(0);
    std::atomic<size_t> intervalCorruptEventCount(0);

    std::thread consumer([&]() {
        uint64_t lastPulseId = 0;

        while (!stopFlag.load()) {
            KMicroscopeData event;
            if (handler.popEvent(event)) {
                eventCount++;
                intervalEventCount++;

                // Check if pulseId is out of order (corrupted)
                if (event.pulseid < lastPulseId) {
                    corruptEventCount++;
                    intervalCorruptEventCount++;

                    int64_t pulseIdDiff = static_cast<int64_t>(event.pulseid) - static_cast<int64_t>(lastPulseId);
                    std::cout << "Corrupt Event! Last PulseId: " << lastPulseId
                            << ", Current PulseId: " << event.pulseid
                            << ", Difference: " << pulseIdDiff << std::endl;
                }
                lastPulseId = event.pulseid;
            }
        }
    });


    // Run for 45 seconds and print interval rates every 3 seconds.
    constexpr double runTimeSec = 90.0;
    constexpr int intervalSec = 3;
    for (int i = 0; i < runTimeSec / intervalSec; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(intervalSec));

        // Get the number of events processed in the last interval
        size_t intervalEvents = intervalEventCount.exchange(0);
        size_t intervalCorruptEvents = intervalCorruptEventCount.exchange(0); // Reset corrupt count

        double intervalRateKHz = intervalEvents / (double)intervalSec / 1000.0;
        double corruptRate = (intervalEvents > 0) ? (100.0 * intervalCorruptEvents / intervalEvents) : 0.0;

        std::cout << "Interval " << (i + 1) * intervalSec << "s: "
                  << intervalRateKHz << " kHz, Corrupt: "
                  << intervalCorruptEvents << " events (" << corruptRate << "%)" << std::endl;
    }


    // Stop the consumer thread.
    stopFlag.store(true);
    consumer.join();

    // Final statistics.
    double totalRate = static_cast<double>(eventCount.load()) / runTimeSec;
    double totalKHz = totalRate / 1000.0;
    std::cout << "\n===== Test Summary =====" << std::endl;
    std::cout << "Processed " << eventCount.load() << " events in "
              << runTimeSec << " seconds. Avg Rate: " << totalKHz << " kHz" << std::endl;
    std::cout << "Total Corrupted Events: " << corruptEventCount.load() << std::endl;
    std::cout << "========================" << std::endl;

    return 0;
}
