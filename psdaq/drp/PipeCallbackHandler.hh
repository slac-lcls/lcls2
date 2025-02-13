#ifndef PIPE_CALLBACK_HANDLER_HH
#define PIPE_CALLBACK_HANDLER_HH

// Standard includes...
#include <queue>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <scTDC/scTDC.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations in the global namespace.
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

    // Constructor and destructor.
    PipeCallbackHandler(int measurementTimeMs,
                        const std::string& iniFilePath,
                        size_t batchSize = 100);
    ~PipeCallbackHandler();

    // Retrieves an event from the event queue; returns false if none available.
    bool popEvent(sc_DldEvent &event);

    // Called (from the callback) to accumulate a batch of events.
    void accumulateEvents(const std::vector<sc_DldEvent>& events);

    // Flushes any pending events into the main event queue.
    void flushPending();

    // Starts the measurement if it hasn't been started already.
    void startMeasurement();

    // Initialize device and open communication pipe
    void init();

private:
    // Member variables...
    int               m_measurementTimeMs;
    size_t            m_batchSize;
    std::string       m_iniFilePath;
    int               m_dd;
    int               m_pd;
    sc_DeviceProperties3 m_sizes;
    sc_pipe_callbacks*   m_cbs;
    sc_pipe_callback_params_t m_params;
    PrivData          m_privData;
    std::queue<sc_DldEvent> m_eventQueue;
    std::mutex      m_queueMutex;
    std::vector<sc_DldEvent> m_pendingBatch;
    std::mutex      m_batchMutex;

    // Flag to indicate whether measurement has been started.
    bool m_measurementStarted;

    // Friend declarations: note the global scope qualifier.
    friend void ::cb_start(void* priv);
    friend void ::cb_end(void* priv);
    friend void ::cb_millis(void* priv);
    friend void ::cb_stat(void* priv, const statistics_t* stat);
    friend void ::cb_tdc_event(void* priv, const sc_TdcEvent* event_array, size_t len);
    friend void ::cb_dld_event(void* priv, const sc_DldEvent* const event_array, size_t event_array_len);
};

} // namespace Drp

#endif  // PIPE_CALLBACK_HANDLER_HH
