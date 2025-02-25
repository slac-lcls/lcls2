#ifndef BUFFERED_DATA_CALLBACK_HANDLER_HH
#define BUFFERED_DATA_CALLBACK_HANDLER_HH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <array>
#include <cstring>
#include <scTDC/scTDC.h>
#include "psalg/utils/SysLog.hh"

#ifdef __cplusplus
extern "C" {
#endif

// Callback declarations for buffered data interface.
void cb_buffered_data(void *priv, const sc_pipe_buf_callback_args* const data);
bool cb_end_of_measurement(void *priv);

#ifdef __cplusplus
}
#endif

namespace Drp {

class BufferedDataCallbackHandler {
public:
    struct PrivData {
        BufferedDataCallbackHandler* handler;
    };

    BufferedDataCallbackHandler(int measurementTimeMs, const std::string &iniFilePath);
    ~BufferedDataCallbackHandler();

    void init();
    void startMeasurement();
    void processBufferedData(const sc_pipe_buf_callback_args* data);

    BufferedDataCallbackHandler(const BufferedDataCallbackHandler&) = delete;
    BufferedDataCallbackHandler& operator=(const BufferedDataCallbackHandler&) = delete;

private:
    int m_measurementTimeMs;
    std::string m_iniFilePath;
    int m_dd;
    int m_pd;
    sc_pipe_buf_callbacks_params_t m_params;
    PrivData m_privData;

    bool m_measurementStarted;
};

} // namespace Drp

#endif // BUFFERED_DATA_CALLBACK_HANDLER_HH
