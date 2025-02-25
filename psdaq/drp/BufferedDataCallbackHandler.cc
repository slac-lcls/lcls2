#include "BufferedDataCallbackHandler.hh"
#include <iostream>
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

namespace Drp {

BufferedDataCallbackHandler::BufferedDataCallbackHandler(int measurementTimeMs,
                                                         const std::string &iniFilePath)
    : m_measurementTimeMs(measurementTimeMs),
      m_iniFilePath(iniFilePath),
      m_measurementStarted(false) {
    // Initialization deferred until init() is called.
}

void BufferedDataCallbackHandler::init() {
    m_dd = sc_tdc_init_inifile(m_iniFilePath.c_str());
    if (m_dd < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(m_dd, error_description);
        throw std::runtime_error("Failed to initialize sc_tdc device: " + std::string(error_description));
    }

    m_privData.handler = this;
    m_params.priv = &m_privData;
    m_params.data = cb_buffered_data;
    m_params.end_of_measurement = cb_end_of_measurement;
    m_params.data_field_selection = SC_DATA_FIELD_DIF1 | SC_DATA_FIELD_DIF2 | SC_DATA_FIELD_TIME | SC_DATA_FIELD_TIME_TAG | SC_DATA_FIELD_START_COUNTER;
    m_params.max_buffered_data_len = 1024; // Arbitrary buffer size
    m_params.dld_events = 1;
    m_params.version = 0;

    m_pd = sc_pipe_open2(m_dd, BUFFERED_DATA_CALLBACKS, &m_params);
    if (m_pd < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(m_pd, error_description);
        throw std::runtime_error("Failed to open buffered data pipe: " + std::string(error_description));
    }
}

void BufferedDataCallbackHandler::startMeasurement() {
    if (!m_measurementStarted) {
        int ret = sc_tdc_start_measure2(m_dd, m_measurementTimeMs);
        logging::debug("BufferedDataCallbackHandler::startMeasurement() measurementTimeMs: %i", m_measurementTimeMs);
        if (ret < 0) {
            char error_description[ERRSTRLEN];
            sc_get_err_msg(ret, error_description);
            throw std::runtime_error("Failed to start measurement: " + std::string(error_description));
        }
        m_measurementStarted = true;
    }
}

BufferedDataCallbackHandler::~BufferedDataCallbackHandler() {
    if (m_pd >= 0) {
        sc_pipe_close2(m_dd, m_pd);
        sc_tdc_deinit2(m_dd);
    }
}

void BufferedDataCallbackHandler::processBufferedData(const sc_pipe_buf_callback_args* data) {
    if (!data || data->data_len == 0) return;

    for (size_t i = 0; i < data->data_len; ++i) {
        std::cout << "dif1: " << data->dif1[i] << ", dif2: " << data->dif2[i]
                  << ", time: " << data->time[i] << ", time_tag: " << data->time_tag[i]
                  << ", start_counter: " << data->start_counter[i] << std::endl;
    }
}

extern "C" {
void cb_buffered_data(void *priv, const sc_pipe_buf_callback_args* const data) {
    auto* handler = static_cast<Drp::BufferedDataCallbackHandler*>(priv);
    if (handler) {
        handler->processBufferedData(data);
    }
}

bool cb_end_of_measurement(void *priv) {
    return true; // Ensures any remaining buffered data is processed immediately.
}
} // extern "C"

} // namespace Drp
