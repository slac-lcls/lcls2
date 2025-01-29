#include "PipeCallbackHandler.hh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace Drp {

PipeCallbackHandler::PipeCallbackHandler(const std::string& configFile)
    : deviceDescriptor_(-1), pipeDescriptor_(-1) {
    sem_init(&semaphore_, 0, 0);
    init(configFile);
}

PipeCallbackHandler::~PipeCallbackHandler() {
    closePipe();
    deinit();
    sem_destroy(&semaphore_);
}

void PipeCallbackHandler::init(const std::string& configFile) {
    deviceDescriptor_ = sc_tdc_init_inifile(configFile.c_str());
    if (deviceDescriptor_ < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(deviceDescriptor_, error_description);
        throw std::runtime_error(std::string("Error initializing TDC: ") + error_description);
    }

    int ret = sc_tdc_get_device_properties(deviceDescriptor_, 3, &sizes_);
    if (ret < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(ret, error_description);
        throw std::runtime_error(std::string("Error getting device properties: ") + error_description);
    }
}

void PipeCallbackHandler::deinit() {
    if (deviceDescriptor_ >= 0) {
        sc_tdc_deinit2(deviceDescriptor_);
        deviceDescriptor_ = -1;
    }
}

void PipeCallbackHandler::openPipe() {
    char* buffer = (char*)calloc(1, sizes_.user_callback_size);
    struct sc_pipe_callbacks* cbs = reinterpret_cast<struct sc_pipe_callbacks*>(buffer);

    PrivData* priv_data = new PrivData();
    setupCallbacks(cbs, priv_data);

    struct sc_pipe_callback_params_t params = { cbs };
    pipeDescriptor_ = sc_pipe_open2(deviceDescriptor_, USER_CALLBACKS, &params);
    if (pipeDescriptor_ < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(pipeDescriptor_, error_description);
        free(buffer);
        throw std::runtime_error(std::string("Error opening pipe: ") + error_description);
    }

    free(buffer);
}

void PipeCallbackHandler::closePipe() {
    if (pipeDescriptor_ >= 0) {
        sc_pipe_close2(deviceDescriptor_, pipeDescriptor_);
        pipeDescriptor_ = -1;
    }
}

void PipeCallbackHandler::setupCallbacks(struct sc_pipe_callbacks* cbs, PrivData* priv_data) {
    cbs->priv = priv_data;
    cbs->start_of_measure = PipeCallbackHandler::startOfMeasurement;
    cbs->end_of_measure = PipeCallbackHandler::endOfMeasurement;
    cbs->tdc_event = PipeCallbackHandler::tdcEvent;
    cbs->dld_event = PipeCallbackHandler::dldEvent;
}

void PipeCallbackHandler::startMeasurement(int duration_ms) {
    openPipe();

    int ret = sc_tdc_start_measure2(deviceDescriptor_, duration_ms);
    if (ret < 0) {
        char error_description[ERRSTRLEN];
        sc_get_err_msg(ret, error_description);
        throw std::runtime_error(std::string("Error starting measurement: ") + error_description);
    }
}

void PipeCallbackHandler::waitForEndOfMeasurement() {
    sem_wait(&semaphore_);
}

void PipeCallbackHandler::startOfMeasurement(void* p) {
    auto* priv_data = static_cast<PrivData*>(p);
    printf("START: cn_measures = %d\n", priv_data->cn_measures);
}

void PipeCallbackHandler::endOfMeasurement(void* p) {
    auto* priv_data = static_cast<PrivData*>(p);
    priv_data->cn_measures++;
    printf("END: cn_measures = %d\n", priv_data->cn_measures);
    auto* handler = reinterpret_cast<PipeCallbackHandler*>(p);
    handler->waitForEndOfMeasurement();
}

void PipeCallbackHandler::tdcEvent(void* priv, const struct sc_TdcEvent* const event_array, size_t event_array_len) {
    auto* priv_data = static_cast<PrivData*>(priv);
    priv_data->cn_tdc_events++;
    printf("TDCEVENT: %d\n", priv_data->cn_tdc_events);
}

void PipeCallbackHandler::dldEvent(void* priv, const struct sc_DldEvent* const event_array, size_t event_array_len) {
    auto* priv_data = static_cast<PrivData*>(priv);
    priv_data->cn_dld_events++;
    printf("DLDEVENT: %d\n", priv_data->cn_dld_events);
}

} // namespace Drp

