#ifndef PIPE_CALLBACK_HANDLER_HH
#define PIPE_CALLBACK_HANDLER_HH

#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include <cstring>  // for memset
#include <scTDC/scTDC.h>
#include "spscqueue.hh"  // Use SPSCQueue

#ifdef __cplusplus
extern "C" {
#endif

void cb_start(void *priv);
void cb_end(void *priv);
void cb_millis(void *priv);
void cb_stat(void *priv, const statistics_t* stat);
void cb_tdc_event(void *priv, const sc_TdcEvent* event_array, size_t len);
void cb_dld_event(void *priv, const sc_DldEvent* const event_array, size_t event_array_len);

#ifdef __cplusplus
}
#endif

namespace Drp {

//-----------------------------------------------------------------------
// KMicroscopeData class: holds up to 16 raw events with the same pulseid.
// Each event record consists of:
//   • 2 bytes for xpos (uint16_t)
//   • 2 bytes for ypos (uint16_t)
//   • 4 bytes for time (uint32_t)
// Additionally, we store a common pulseid and the number of events (count).
//-----------------------------------------------------------------------
class KMicroscopeData {
public:
    static const size_t MAX_EVENTS = 16; // Maximum number of raw events to save.

    // Data fields: fixed-size arrays to store event data.
    uint16_t xpos[MAX_EVENTS];
    uint16_t ypos[MAX_EVENTS];
    uint32_t time[MAX_EVENTS];
    uint64_t pulseid; // Common pulseid (from time_tag)
    size_t   count;   // Number of valid events stored

    // Default constructor: initialize arrays to zero.
    KMicroscopeData() : pulseid(0), count(0) {
        memset(xpos, 0, sizeof(xpos));
        memset(ypos, 0, sizeof(ypos));
        memset(time,  0, sizeof(time));
    }

    // addEvent() appends one raw event's data.
    // We no longer pass in the pulseid because it is set and checked externally.
    void addEvent(uint16_t x, uint16_t y, uint32_t t) {
        if (count < MAX_EVENTS) {
            xpos[count] = x;
            ypos[count] = y;
            time[count] = t;
            count++;
        }
    }
};

//-----------------------------------------------------------------------
// PipeCallbackHandler class declaration
//-----------------------------------------------------------------------
class PipeCallbackHandler {
public:
    struct PrivData {
        int   cn_measures;
        int   cn_tdc_events;
        int   cn_dld_events;
        double total_time;
        size_t dld_event_size;        // Size (in bytes) of a DLD event.
        PipeCallbackHandler* handler; // Back–pointer to this handler.
    };

    PipeCallbackHandler(int measurementTimeMs,
                        const std::string& iniFilePath,
                        size_t queueCapacity);
    ~PipeCallbackHandler();

    void init();
    void startMeasurement();
    // popEvent returns a completed KMicroscopeData. A KMicroscopeData is considered complete
    // when a new pulseid is encountered. The in-progress event is not popped.
    bool popEvent(KMicroscopeData &event);

    // Delete copy constructor and copy-assignment operator
    PipeCallbackHandler(const PipeCallbackHandler&) = delete;
    PipeCallbackHandler& operator=(const PipeCallbackHandler&) = delete;

private:
    int m_measurementTimeMs;
    std::string m_iniFilePath;
    int m_dd;
    int m_pd;
    sc_DeviceProperties3 m_sizes;
    sc_pipe_callbacks* m_cbs;
    sc_pipe_callback_params_t m_params;
    PrivData m_privData;

    bool m_measurementStarted;

    // Member used to accumulate raw sc_DldEvent records until a pulseid change is seen.
    KMicroscopeData m_currentEvent; // in-progress KMicroscopeData
    bool m_hasCurrentEvent;         // flag indicating an event is in progress

    // Replace the standard queue with a lock-free SPSCQueue.
    SPSCQueue<KMicroscopeData> m_eventQueue;

    // Add the mutex here to protect m_currentEvent
    mutable std::mutex m_eventMutex;

    // Process a single raw sc_DldEvent.
    void processScDldEvent(const sc_DldEvent* obj);

    friend void ::cb_start(void* priv);
    friend void ::cb_end(void* priv);
    friend void ::cb_millis(void* priv);
    friend void ::cb_stat(void* priv, const statistics_t* stat);
    friend void ::cb_tdc_event(void* priv, const sc_TdcEvent* event_array, size_t len);
    friend void ::cb_dld_event(void* priv, const sc_DldEvent* const event_array, size_t event_array_len);
};

} // namespace Drp

#endif // PIPE_CALLBACK_HANDLER_HH
