#include "gpuUtils.hh"

#include "psalg/utils/SysLog.hh"

#include <string>


using logging = psalg::SysLog;
using namespace Drp::Gpu;


// -------------------------------------------------------------------

bool CudaContext::init(int device, bool quiet) {

    int devCount = 0;
    if (chkError(cudaGetDeviceCount(&devCount)))
        return false;
    logging::debug("Total GPU devices: %d", devCount);
    if (devCount <= 0) {
        logging::error("No GPU devices available!");
        return false;
    }

    if (!quiet)  listDevices();

    device = device < 0 ? 0 : device;
    if (devCount <= device) {
        logging::error("Invalid GPU device number %d! There are only %d devices available", device, devCount);
        return false;
    }
    _devNo = device;

    // Actually get the device...
    CUresult status;
    if ((status = cuDeviceGet(&device_, device)) != CUDA_SUCCESS) {
        logging::error("Failed to get GPU device %d: code=%d", device, status);
        return false;
    }

    // Spew device name
    char name[256];
    if (chkError(cuDeviceGetName(name, sizeof(name), device_)))
        return false;
    logging::info("Selected GPU device %d: %s", device, name);

    cudaDeviceProp deviceProp;
    chkError(cudaGetDeviceProperties(&deviceProp, device_));
    logging::info("Compute Capability: %d.%d", deviceProp.major, deviceProp.minor);

    // Report memory totals
    size_t global_mem = 0;
    if (chkError(cuDeviceTotalMem(&global_mem, device_)))
        return false;
    logging::info("Global memory: %zu MB", global_mem >> 20);
    if (global_mem > (size_t)4 << 30)
        logging::debug("64-bit Memory Address support");

    // Create context
#if CUDA_VERSION >= 13000
    constexpr unsigned flags = CU_CTX_SCHED_SPIN; // @todo: Revisit
    if (chkError(cuCtxCreate(&context_, NULL, flags, device_)))
#else
    if (chkError(cuCtxCreate(&context_, 0, device_)))
#endif
        return false;

    return true;
}

void CudaContext::listDevices() {
    int devs = 0;
    CUresult status;
    if ((status = cuDeviceGetCount(&devs)) != CUDA_SUCCESS) {
        logging::error("Unable to get GPU device count");
        return;
    }

    for (int i = 0; i < devs; ++i) {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            logging::error("Unable to get GPU device %d", i);
            continue;
        }
        char name[256];
        if (cuDeviceGetName(name, sizeof(name), dev) != CUDA_SUCCESS) {
            logging::error("Unable to get name of GPU device %d", i);
            continue;
        }
        logging::info("%d: %s", i, name);
    }
}

// -------------------------------------------------------------------

void CoreRegisters::initialize(bool sim, void* regs)
{
  if (!sim) _fwRegs = std::make_unique<GpuAsyncCoreRegs>(regs);
  else      _swRegs = static_cast<uint32_t*>(regs);
}

// -------------------------------------------------------------------

int CudaContext::getAttribute(CUdevice_attribute attr)
{
    int out;
    if (cuDeviceGetAttribute(&out, attr, device_) == CUDA_SUCCESS)
        return out;
    return 0;
}

// -------------------------------------------------------------------

bool Drp::Gpu::checkError(CUresult status, const char* const func, const char* const file,
                          const int line, const bool crash, const char* const msg)
{
    if (status != CUDA_SUCCESS) {
        const char* perrstr = 0;
        CUresult ok         = cuGetErrorString(status, &perrstr);
        const char* perrnam = 0;
        CUresult ok2        = cuGetErrorName(status, &perrnam);
        const char* message = msg ? msg : ""; // Just in case, but msg is never 0
        if (ok == CUDA_SUCCESS && ok2 == CUDA_SUCCESS) {
            if (perrstr) {
                logging::error("%s:%d:  %s (%i): '%s' %s",
                               file, line, perrnam, status, perrstr, message);
            } else {
                logging::error("%s:%d:  %s (%i): unknown error %s",
                               file, line, perrnam, status, message);
            }
        } else {
            logging::error("%s:%d:  status %i: unknown error %s",
                           file, line, status, message);
        }
        if (crash)  abort();
        return true;
    }
    return false;
}

bool Drp::Gpu::checkError(cudaError status, const char* const func, const char* const file,
                          const int line, const bool crash, const char* const msg)
{
    if (status != cudaSuccess) {
        logging::error("%s:%d:  %s (%i): '%s' %s",
                       file, line, cudaGetErrorName(status), status,
                       cudaGetErrorString(status), msg ? msg : "");
        if (crash)  abort();
        return true;
    }
    return false;
}

// -------------------------------------------------------------------

DmaTgt_t Drp::Gpu::dmaTgtGet(CoreRegisters& coreRegs)
{
    DmaTgt_t tgt;
    switch (coreRegs.axisDeMuxSelect()) {
        case TGT_CPU:  tgt = TGT_CPU;  break;
        case TGT_GPU:  tgt = TGT_GPU;  break;
        default:       tgt = TGT_ERR;  break;
    }
    return tgt;
}

void Drp::Gpu::dmaTgtSet(CoreRegisters& coreRegs, DmaTgt_t tgt)
{
    coreRegs.setAxisDeMuxSelect(tgt);
}

/** Function to reset the DMA buffer index */
void Drp::Gpu::dmaIdxReset(CoreRegisters& coreRegs)
{
    coreRegs.setWriteEnable(0);
    coreRegs.setWriteEnable(1);
}

// -------------------------------------------------------------------
