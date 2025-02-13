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
    static const size_t MAX_EVENTS = 16; // Maximum number of raw events to save.

    // Data fields: xpos and ypos are stored as uint16_t, time as uint32_t.
    std::array<uint16_t, MAX_EVENTS> xpos;
    std::array<uint16_t, MAX_EVENTS> ypos;
    std::array<uint32_t, MAX_EVENTS> time;
    uint64_t pulseid; // Common pulse id (from time_tag)
    size_t count;     // Number of valid events stored

    // Fixed-size buffer (128 bytes) for building the payload of 8 bytes x 16 MAX_EVENTS.
    std::array<uint8_t, 128> m_buffer;

    DrpEvent() : pulseid(0), count(0) { }

    // payload() encodes the stored data into m_buffer.
    // Each record is 8 bytes: 2 bytes for xpos, 2 for ypos, 4 for time.
    // The method returns a pointer to m_buffer.data(), and the payload
    // is exactly count * 8 bytes long.
    uint8_t* payload() {
        constexpr size_t recordSize = 8; // 2 + 2 + 4 = 8 bytes per record.
        size_t totalSize = count * recordSize;
        if (totalSize > m_buffer.size()) {
            throw std::runtime_error("DrpEvent payload exceeds maximum allowed size");
        }
        for (size_t i = 0; i < count; i++) {
            size_t offset = i * recordSize;
            // Write xpos[i] in little-endian order.
            m_buffer[offset]     = static_cast<uint8_t>(xpos[i] & 0xFF);
            m_buffer[offset + 1] = static_cast<uint8_t>((xpos[i] >> 8) & 0xFF);
            // Write ypos[i] in little-endian order.
            m_buffer[offset + 2] = static_cast<uint8_t>(ypos[i] & 0xFF);
            m_buffer[offset + 3] = static_cast<uint8_t>((ypos[i] >> 8) & 0xFF);
            // Write time[i] in little-endian order.
            m_buffer[offset + 4] = static_cast<uint8_t>(time[i] & 0xFF);
            m_buffer[offset + 5] = static_cast<uint8_t>((time[i] >> 8) & 0xFF);
            m_buffer[offset + 6] = static_cast<uint8_t>((time[i] >> 16) & 0xFF);
            m_buffer[offset + 7] = static_cast<uint8_t>((time[i] >> 24) & 0xFF);
        }
        return m_buffer.data();
    }

    // payloadSize() returns the total number of bytes in the payload.
    unsigned payloadSize() const {
        return static_cast<unsigned>(count * recordSize());
    }

private:
    // Helper function to return record size (in bytes).
    constexpr size_t recordSize() const { return 8; }
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
