#include "BufferedDataCallbackHandler.hh"
#include <iostream>
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

namespace Drp {

BufferedDataCallbackHandler::BufferedDataCallbackHandler(int measurementTimeMs, const std::string &iniFilePath)
    : m_measurementTimeMs(measurementTimeMs),
      m_iniFilePath(iniFilePath),
      m_measurementStarted(false) {

    // Create or open shared memory object
    shm_fd = shm_open("/shared_event_data", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open failed");
        throw std::runtime_error("Failed to create shared memory.");
    }
    std::cerr << "Shared memory created successfully.\n";

    // Set size of shared memory
    if (ftruncate(shm_fd, sizeof(SharedEventData)) == -1) {
        perror("ftruncate failed");
        throw std::runtime_error("Failed to set size of shared memory.");
    }
    std::cerr << "Shared memory size set successfully.\n";

    // Map shared memory to process address space
    shared_event = static_cast<SharedEventData*>(mmap(
        nullptr, sizeof(SharedEventData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0
    ));

    if (shared_event == MAP_FAILED) {
        perror("mmap failed");
        throw std::runtime_error("Failed to map shared memory.");
    }

    // Initialize shared memory and verify its allocation
    *shared_event = SharedEventData();  // Correctly initialize struct
    shared_event->is_valid = false;

    std::cerr << "Shared memory successfully mapped at " << static_cast<void*>(shared_event) << "\n";
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

    // Unmap and close shared memory
    if (shared_event) {
        munmap(shared_event, sizeof(SharedEventData));
    }
    if (shm_fd >= 0) {
        close(shm_fd);
    }
    shm_unlink("/shared_event_data");  // Ensure it's cleaned up
}

void BufferedDataCallbackHandler::processBufferedData(const sc_pipe_buf_callback_args* data) {
    if (!data || data->data_len == 0 || data->data_len > 1024) {
        std::cerr << "ERROR: Received invalid event! Skipping...\n";
        return;
    }

    // Validate source pointers before copying
    if (!data->dif1 || !data->dif2 || !data->time || !data->time_tag || !data->start_counter) {
        std::cerr << "ERROR: Incoming data contains NULL pointers! Skipping event.\n";
        return;
    }

    // Validate shared memory before copying
    if (shared_event == nullptr) {
        std::cerr << "ERROR: shared_event is NULL! Shared memory may not be mapped correctly.\n";
        return;
    }

    try {
        // Corrected type sizes for safe copying
        std::memcpy(shared_event->dif1.data(), data->dif1, data->data_len * sizeof(uint32_t));  // Now 32-bit
        std::memcpy(shared_event->dif2.data(), data->dif2, data->data_len * sizeof(uint32_t));  // Now 32-bit
        std::memcpy(shared_event->time.data(), data->time, data->data_len * sizeof(uint64_t));  // Now 64-bit
        std::memcpy(shared_event->time_tag.data(), data->time_tag, data->data_len * sizeof(uint32_t));  // Now 32-bit
        std::memcpy(shared_event->start_counter.data(), data->start_counter, data->data_len * sizeof(uint64_t));  // Now 64-bit

        shared_event->data_len = data->data_len;
        shared_event->is_valid = true;

        std::cerr << "Successfully copied event. Data length: " << shared_event->data_len << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception in memcpy: " << e.what() << "\n";
    }
}


bool BufferedDataCallbackHandler::getLatestEvent(SharedEventData& event) {
    if (!shared_event->is_valid) {
        return false;
    }

    memcpy(&event, shared_event, sizeof(SharedEventData));
    return true;
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
