#ifndef PIPE_CALLBACK_HANDLER_HH
#define PIPE_CALLBACK_HANDLER_HH

#include <queue>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include <scTDC/scTDC.h>

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
// DrpEvent class: groups up to 16 raw events with the same pulseid.
// The data are now stored in fixed–size arrays using integer types.
// Each record will be encoded into 8 bytes:
//  • 2 bytes for xpos (uint16_t)
//  • 2 bytes for ypos (uint16_t)
//  • 4 bytes for time (uint32_t)
//-----------------------------------------------------------------------
class DrpEvent {
public:
    static const size_t MAX_EVENTS = 16; // maximum number of raw events to save
    std::array<uint16_t, MAX_EVENTS> xpos; // changed from double to uint16_t
    std::array<uint16_t, MAX_EVENTS> ypos; // changed from double to uint16_t
    std::array<uint32_t, MAX_EVENTS> time;   // changed from double to uint32_t
    uint64_t pulseid; // common pulse id (from time_tag)
    size_t count;     // number of events stored

    DrpEvent() : pulseid(0), count(0) { }

    // payload() builds a contiguous block of memory where each record is:
    //   [xpos (2 bytes), ypos (2 bytes), time (4 bytes)]
    // Caller is responsible for freeing the returned memory.
    uint8_t* payload() const {
        size_t recordSize = 2 + 2 + 4; // 8 bytes per record.
        size_t totalSize = count * recordSize;
        uint8_t* buf = new uint8_t[totalSize];
        for (size_t i = 0; i < count; i++) {
            size_t offset = i * recordSize;
            // Write xpos[i] in little-endian order.
            buf[offset]     = static_cast<uint8_t>(xpos[i] & 0xFF);
            buf[offset + 1] = static_cast<uint8_t>((xpos[i] >> 8) & 0xFF);
            // Write ypos[i] in little-endian order.
            buf[offset + 2] = static_cast<uint8_t>(ypos[i] & 0xFF);
            buf[offset + 3] = static_cast<uint8_t>((ypos[i] >> 8) & 0xFF);
            // Write time[i] in little-endian order.
            buf[offset + 4] = static_cast<uint8_t>(time[i] & 0xFF);
            buf[offset + 5] = static_cast<uint8_t>((time[i] >> 8) & 0xFF);
            buf[offset + 6] = static_cast<uint8_t>((time[i] >> 16) & 0xFF);
            buf[offset + 7] = static_cast<uint8_t>((time[i] >> 24) & 0xFF);
        }
        return buf;
    }

    // payloadSize() returns the size (in bytes) of the payload.
    unsigned payloadSize() const {
        return static_cast<unsigned>(count * 8);
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
                        size_t batchSize = 100);
    ~PipeCallbackHandler();

    void init();
    void startMeasurement();
    // popEvent returns a completed DrpEvent. A DrpEvent is considered complete
    // when a new pulseid is encountered. The in-progress event is not popped.
    bool popEvent(DrpEvent &event);
    void flushPending();
    void accumulateEvents(const std::vector<DrpEvent>& events);

private:
    int m_measurementTimeMs;
    size_t m_batchSize;
    std::string m_iniFilePath;
    int m_dd;
    int m_pd;
    sc_DeviceProperties3 m_sizes;
    sc_pipe_callbacks* m_cbs;
    sc_pipe_callback_params_t m_params;
    PrivData m_privData;

    std::queue<DrpEvent> m_eventQueue;
    std::mutex m_queueMutex;
    std::vector<DrpEvent> m_pendingBatch;
    std::mutex m_batchMutex;

    bool m_measurementStarted;

    // Members used to accumulate raw sc_DldEvent records until a pulseid change is seen.
    DrpEvent m_currentEvent; // in–progress DrpEvent
    bool m_hasCurrentEvent;  // flag indicating an event is in progress

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
