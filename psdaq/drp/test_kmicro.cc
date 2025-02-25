#include <iostream>
#include <thread>
#include <chrono>
#include "BufferedDataCallbackHandler.hh"

using namespace Drp;

int main(int argc, char* argv[]) {
    // Default parameters.
    int measurementTimeMs = 1000;
    std::string iniFilePath = "tdc_gpx3.ini";

    // Command-line arguments override defaults.
    if (argc > 1)
        measurementTimeMs = std::stoi(argv[1]);
    if (argc > 2)
        iniFilePath = argv[2];

    std::cout << "Using options:" << std::endl;
    std::cout << "  Measurement Time (ms): " << measurementTimeMs << std::endl;
    std::cout << "  INI File Path: " << iniFilePath << std::endl;

    // Create and initialize the handler.
    BufferedDataCallbackHandler handler(measurementTimeMs, iniFilePath);
    handler.init();
    handler.startMeasurement();

    // Let the test run for a fixed duration to capture and print events.
    constexpr int runTimeSec = 5;
    std::cout << "Running for " << runTimeSec << " seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(runTimeSec));

    std::cout << "Test complete." << std::endl;
    return 0;
}
