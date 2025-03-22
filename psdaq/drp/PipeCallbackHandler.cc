#include "PipeCallbackHandler.hh"
#include <thread>
#include <chrono>
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

namespace Drp {

namespace {
    // Helper function to validate that 'value' is a power of 2.
    inline size_t validatePowerOf2(size_t value, const std::string &name) {
        if ((value & (value - 1)) != 0) {
            throw std::runtime_error(name + " must be a power of 2");
        }
        return value;
    }
}

// Constructor that takes separate parameters for flushing and queue capacity.
PipeCallbackHandler::PipeCallbackHandler(int measurementTimeMs,
                                         const std::string& iniFilePath,
                                         size_t queueCapacity)
    : m_measurementTimeMs(measurementTimeMs),
      m_iniFilePath(iniFilePath),
      m_measurementStarted(false),
      m_hasCurrentEvent(false),
      m_eventQueue(validatePowerOf2(queueCapacity, "Queue capacity"))
{
    // SC initialization is deferred until init() is called.
}

void PipeCallbackHandler::init(){
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

    // Allocate and initialize the callback structure.
    char* buffer = static_cast<char*>(calloc(1, m_sizes.user_callback_size));
    if (!buffer) {
        throw std::runtime_error("Failed to allocate memory for callback structure");
    }
    m_cbs = reinterpret_cast<sc_pipe_callbacks*>(buffer);
    m_cbs->priv = &m_privData;
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
}

void PipeCallbackHandler::startMeasurement() {
    // Start the measurement only if it hasn't been started before.
    if (!m_measurementStarted) {
        int ret = sc_tdc_start_measure2(m_dd, m_measurementTimeMs);
        logging::debug("PipeCallbackHandler::startMeasurement() measurementTimeMs: %i", m_measurementTimeMs);
        if (ret < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(ret, error_description);
            printf("error! code: %d, message: %s\n", ret, error_description);
            throw std::runtime_error("Failed to start measurement");
        }
        m_measurementStarted = true;
    }
}

PipeCallbackHandler::~PipeCallbackHandler() {
    if (m_pd >= 0) {
        sc_pipe_close2(m_dd, m_pd);
        m_pd = -1; // Mark as closed.
        sc_tdc_deinit2(m_dd);
    }
    m_measurementStarted = false;
}

bool PipeCallbackHandler::popEvent(KMicroscopeData &event) {
    // Use the lock-free try_pop() method from SPSCQueue.
    return m_eventQueue.try_pop(event);
}

//------------------------------------------------------------------------------
// processScDldEvent: Processes a single raw sc_DldEvent and updates the in-progress KMicroscopeData.
// If a new pulseid is encountered, the current event is complete, stored in the queue,
// and a new KMicroscopeData is started.
//------------------------------------------------------------------------------
void PipeCallbackHandler::processScDldEvent(const sc_DldEvent* obj) {
    uint64_t newPulse = obj->time_tag;

    // Check if the pulse ID is not advancing correctly
    if (m_hasCurrentEvent && newPulse < m_currentEvent.pulseid) {
        int64_t pulseDiff = static_cast<int64_t>(newPulse) - static_cast<int64_t>(m_currentEvent.pulseid);
        printf("Unexpected PulseId behavior! Last PulseId: %lu, Current PulseId: %lu, Difference: %ld\n",
                         m_currentEvent.pulseid, newPulse, pulseDiff);
    }

    if (!m_hasCurrentEvent) {
        m_currentEvent = KMicroscopeData();
        m_currentEvent.pulseid = newPulse;
        m_hasCurrentEvent = true;
    }

    if (m_hasCurrentEvent && m_currentEvent.pulseid != newPulse) {
        m_eventQueue.push(m_currentEvent);
        m_currentEvent = KMicroscopeData();
        m_currentEvent.pulseid = newPulse;
    }

    if (m_currentEvent.count < KMicroscopeData::MAX_EVENTS) {
        m_currentEvent.addEvent(obj->dif1, obj->dif2, obj->sum);
    }
}

} // namespace Drp

//------------------------------------------------------------------------------
// Callback Function Definitions (with C linkage)
//------------------------------------------------------------------------------
extern "C" {

void cb_start(void *priv) {
    // Called at the start of a measurement.
}

void cb_end(void *priv) {
    // Called at the end of a measurement.
}

void cb_millis(void *priv) {
    // Called every millisecond.
}

void cb_stat(void *priv, const statistics_t* stat) {
    // Process statistics if desired.
}

void cb_tdc_event(void *priv, const sc_TdcEvent* event_array, size_t len) {
    // Process TDC events if desired.
}

void cb_dld_event(void *priv, const sc_DldEvent* const event_array, size_t event_array_len) {
    Drp::PipeCallbackHandler::PrivData* pData =
        static_cast<Drp::PipeCallbackHandler::PrivData*>(priv);
    pData->cn_dld_events++;

    const char* buffer = reinterpret_cast<const char*>(event_array);
    for (size_t j = 0; j < event_array_len; ++j) {
        const sc_DldEvent* obj = reinterpret_cast<const sc_DldEvent*>(buffer + j * pData->dld_event_size);
        pData->handler->processScDldEvent(obj);
    }
}

} // extern "C"
