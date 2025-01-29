#pragma once

#include <semaphore.h>
#include <scTDC.h>
#include <inttypes.h>
#include <string>
#include <memory>

namespace Drp {

// Struct to hold private data for the callbacks
struct PrivData {
    int cn_measures = 0;
    int cn_tdc_events = 0;
    int cn_dld_events = 0;
    double total_time = 0.0;
};

// Class to encapsulate the pipe callbacks and related operations
class PipeCallbackHandler {
public:
    PipeCallbackHandler(const std::string& configFile);
    ~PipeCallbackHandler();

    void setupCallbacks(struct sc_pipe_callbacks* cbs, PrivData* priv_data);
    void startMeasurement(int duration_ms);
    void waitForEndOfMeasurement();

private:
    void init(const std::string& configFile);
    void deinit();
    void openPipe();
    void closePipe();

    // Callback functions
    static void startOfMeasurement(void* p);
    static void endOfMeasurement(void* p);
    static void tdcEvent(void* priv, const struct sc_TdcEvent* const event_array, size_t event_array_len);
    static void dldEvent(void* priv, const struct sc_DldEvent* const event_array, size_t event_array_len);

    // Members
    sem_t semaphore_;                  // Semaphore for synchronization
    int deviceDescriptor_;             // Descriptor for the TDC device
    int pipeDescriptor_;               // Descriptor for the data pipe
    struct sc_DeviceProperties3 sizes_;// Device properties
};

} // namespace Drp
