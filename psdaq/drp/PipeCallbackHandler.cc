#include "PipeCallbackHandler.hh"
#include <thread>
#include <chrono>
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

namespace Drp {

PipeCallbackHandler::PipeCallbackHandler(int measurementTimeMs,
                                         const std::string& iniFilePath,
                                         size_t batchSize)
    : m_measurementTimeMs(measurementTimeMs),
      m_batchSize(batchSize),
      m_iniFilePath(iniFilePath),
      m_measurementStarted(false)   // new flag: measurement not yet started
{
    logging::info("PipeCallbackHandler::constructor");

}

void PipeCallbackHandler::startMeasurement() {
    // Start the measurement only if it hasn't been started before.
    if (!m_measurementStarted) {
        // Initialize the scTDC device using the specified INI file.
        m_dd = sc_tdc_init_inifile(m_iniFilePath.c_str());
        if (m_dd < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(m_dd, error_description);
            printf("error! code: %d, message: %s\n", m_dd, error_description);
            throw std::runtime_error("Failed to initialize sc_tdc device");
        }

        // Get device properties (assume device type “3” is expected).
        int ret = sc_tdc_get_device_properties(m_dd, 3, &m_sizes);
        if (ret < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(ret, error_description);
            printf("error! code: %d, message: %s\n", ret, error_description);
            throw std::runtime_error("Failed to get device properties");
        }

        // Initialize private callback data.
        m_privData.cn_measures   = 0;
        m_privData.cn_tdc_events = 0;
        m_privData.cn_dld_events = 0;
        m_privData.total_time    = 0.0;
        m_privData.dld_event_size = m_sizes.dld_event_size;
        m_privData.handler       = this;

        // Allocate a buffer for the callback structure.
        char* buffer = static_cast<char*>(calloc(1, m_sizes.user_callback_size));
        if (!buffer) {
            throw std::runtime_error("Failed to allocate memory for callback structure");
        }
        m_cbs = reinterpret_cast<sc_pipe_callbacks*>(buffer);
        m_cbs->priv = &m_privData;

        // Assign our callback functions.
        m_cbs->start_of_measure    = cb_start;
        m_cbs->end_of_measure      = cb_end;
        m_cbs->millisecond_countup = cb_millis;
        m_cbs->statistics          = cb_stat;
        m_cbs->tdc_event           = cb_tdc_event;
        m_cbs->dld_event           = cb_dld_event;
        m_params.callbacks = m_cbs;

        // Open the pipe for communication.
        m_pd = sc_pipe_open2(m_dd, USER_CALLBACKS, &m_params);
        if (m_pd < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(m_pd, error_description);
            printf("error! code: %d, message: %s\n", m_pd, error_description);
            free(buffer);
            throw std::runtime_error("Failed to open pipe");
        }
        free(buffer);

        // Start the measurement using the provided measurement time.
        ret = sc_tdc_start_measure2(m_dd, m_measurementTimeMs);
        if (ret < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(ret, error_description);
            printf("error! code: %d, message: %s\n", ret, error_description);
            throw std::runtime_error("Failed to start measurement");
        }
        m_measurementStarted = true;
    }
}

void PipeCallbackHandler::closePipe() {
    // Close the pipe only if the measurement was started and the pipe is open.
    if (m_measurementStarted && m_pd >= 0) {
        sc_pipe_close2(m_dd, m_pd);
        m_pd = -1; // Mark as closed.
        sc_tdc_deinit2(m_dd);
        m_measurementStarted = false;
    }
}

PipeCallbackHandler::~PipeCallbackHandler() {
    // (Optional) Flush any remaining partial batch into the main queue.
    flushPending();
    closePipe();
}

bool PipeCallbackHandler::popEvent(sc_DldEvent &event) {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_eventQueue.empty()) return false;
    event = m_eventQueue.front();
    m_eventQueue.pop();
    return true;
}

void PipeCallbackHandler::accumulateEvents(const std::vector<sc_DldEvent>& events) {
    std::lock_guard<std::mutex> lock(m_batchMutex);
    // Insert the incoming events into the pending batch.
    m_pendingBatch.insert(m_pendingBatch.end(), events.begin(), events.end());
    // Flush full batches.
    while (m_pendingBatch.size() >= m_batchSize) {
        // Prepare a full batch.
        std::vector<sc_DldEvent> fullBatch(m_pendingBatch.begin(), m_pendingBatch.begin() + m_batchSize);
        {
            std::lock_guard<std::mutex> qlock(m_queueMutex);
            for (const auto &ev : fullBatch) {
                m_eventQueue.push(ev);
            }
        }
        // Remove the flushed events from the accumulation buffer.
        m_pendingBatch.erase(m_pendingBatch.begin(), m_pendingBatch.begin() + m_batchSize);
    }
}

void PipeCallbackHandler::flushPending() {
    std::lock_guard<std::mutex> lock(m_batchMutex);
    if (!m_pendingBatch.empty()) {
        std::lock_guard<std::mutex> qlock(m_queueMutex);
        for (const auto &ev : m_pendingBatch) {
            m_eventQueue.push(ev);
        }
        m_pendingBatch.clear();
    }
}

} // namespace Drp

// -------------------------------------------------------------------------
// Callback Function Definitions (with C linkage, in the global namespace)
// -------------------------------------------------------------------------
extern "C" {

void cb_start(void *priv) {
    // Called at the start of a measurement.
    // printf("Measurement started.\n");
}

void cb_end(void *priv) {
    // Called at the end of a measurement.
    // printf("Measurement ended.\n");
}

void cb_millis(void *priv) {
    // Called every millisecond.
    // printf("Millisecond tick.\n");
}

void cb_stat(void *priv, const statistics_t* stat) {
    // Process statistics if desired.
    // For now, do nothing.
}

void cb_tdc_event(void *priv, const sc_TdcEvent* event_array, size_t len) {
    // Process TDC events if desired.
    // printf("TDC event callback.\n");
}

void cb_dld_event(void *priv, const sc_DldEvent* const event_array, size_t event_array_len) {
    // Increase the DLD event counter.
    Drp::PipeCallbackHandler::PrivData* pData = static_cast<Drp::PipeCallbackHandler::PrivData*>(priv);
    pData->cn_dld_events++;
    // The event_array is a contiguous buffer where each event’s size is given by pData->dld_event_size.
    const char* buffer = reinterpret_cast<const char*>(event_array);
    // Build a local batch of events.
    std::vector<sc_DldEvent> localBatch;
    localBatch.reserve(event_array_len);
    for (size_t j = 0; j < event_array_len; ++j) {
        const sc_DldEvent* obj = reinterpret_cast<const sc_DldEvent*>(buffer + j * pData->dld_event_size);
        // (Optional: Process individual event fields if needed.)
        localBatch.push_back(*obj);
    }
    // Accumulate the events; they will be flushed to the main queue only when a full batch is reached.
    pData->handler->accumulateEvents(localBatch);
}

} // extern "C"
