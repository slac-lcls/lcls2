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
#include <mutex>
#include <fcntl.h>      // For O_* constants
#include <sys/mman.h>   // For shared memory
#include <unistd.h>     // For ftruncate
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

struct SharedEventData {
    size_t data_len;
    std::array<uint32_t, 1024> dif1;           // Corrected to match `unsigned*`
    std::array<uint32_t, 1024> dif2;           // Corrected to match `unsigned*`
    std::array<uint64_t, 1024> time;           // Corrected to match `unsigned long long*`
    std::array<uint32_t, 1024> time_tag;       // Corrected to match `unsigned*`
    std::array<uint64_t, 1024> start_counter;  // Corrected to match `unsigned long long*`
    bool is_valid;

    SharedEventData() : data_len(0), is_valid(false) {}  // Default constructor
};

class BufferedDataCallbackHandler {
public:
    struct PrivData {
        BufferedDataCallbackHandler* handler;
    };
    BufferedDataCallbackHandler(int measurementTimeMs, const std::string &iniFilePath);
    ~BufferedDataCallbackHandler();

    void init();
    void startMeasurement();
    bool getLatestEvent(SharedEventData& event);  // Get data from shared memory
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

    // Shared Memory
    int shm_fd;
    SharedEventData* shared_event;
};

} // namespace Drp

#endif // BUFFERED_DATA_CALLBACK_HANDLER_HH
