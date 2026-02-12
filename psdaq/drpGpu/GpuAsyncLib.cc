#include <string>

#include "GpuAsyncLib.hh"

#include "psdaq/aes-stream-drivers/GpuAsyncRegs.h"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


// -------------------------------------------------------------------

static std::string errorString(CUresult res) {
    const char* ptr = nullptr;
    std::string s;
    cuGetErrorName(res, &ptr);
    s = ptr;
    return s;
}

static std::string errorString(cudaError_t res) {
    return cudaGetErrorName(res);
}

// -------------------------------------------------------------------

DataDev::DataDev(const char* path) {
    fd_ = open(path, O_RDWR);
    if (fd_ < 0) {
        logging::critical("Error opening %s: %m", path);
        abort();
    }
}

// -------------------------------------------------------------------

CudaContext::CudaContext()
{
    chkFatal(cuInit(0));
}

bool CudaContext::init(int device, bool quiet) {

    int devs = 0;
    if (chkError(cuDeviceGetCount(&devs)))
        return false;
    logging::debug("Total GPU devices: %d", devs);
    if (devs <= 0) {
        logging::error("No GPU devices available!");
        return false;
    }

    device = device < 0 ? 0 : device;
    if (devs <= device) {
        logging::error("Invalid GPU device number %d! There are only %d devices available", device, devs);
        return false;
    }
    _devNo = device;

    // Actually get the device...
    CUresult status;
    if ((status = cuDeviceGet(&device_, device)) != CUDA_SUCCESS) {
        logging::error("Could not get GPU device! code=%d", status);
        return false;
    }

    // Spew device name
    char name[256];
    if (chkError(cuDeviceGetName(name, sizeof(name), device_)))
        return false;
    logging::info("Selected GPU device: %s", name);

    cudaDeviceProp deviceProp;
    chkError(cudaGetDeviceProperties(&deviceProp, device_));
    logging::info("Compute Capability: %d.%d", deviceProp.major, deviceProp.minor);

    // Report memory totals
    size_t global_mem = 0;
    if (chkError(cuDeviceTotalMem(&global_mem, device_)))
        return false;
    logging::debug("Global memory: %zu MB", global_mem >> 20);
    if (global_mem > (size_t)4 << 30)
        logging::debug("64-bit Memory Address support");

    // Create context
#if CUDA_VERSION >= 13000
    if (chkError(cuCtxCreate(&context_, NULL, 0, device_)))
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

int CudaContext::getAttribute(CUdevice_attribute attr) {
    int out;
    if (cuDeviceGetAttribute(&out, attr, device_) == CUDA_SUCCESS)
        return out;
    return 0;
}


// -------------------------------------------------------------------

bool checkError(CUresult status, const char* const func, const char* const file,
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

bool checkError(cudaError status, const char* const func, const char* const file,
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

//-----------------------------------------------------------------------------

int gpuInitBufferState(GpuBufferState_t* b, int fd, size_t bufSize)
{
    // Allocate buffers on the GPU
    if (gpuMapFpgaMem(&b->bwrite, fd, 0, bufSize, 1) != 0) {
        logging::error("gpuMapFpgaMem: write: %m");
        return -1;
    }

    if (gpuMapFpgaMem(&b->bread, fd, 0, bufSize, 0) != 0) {
        logging::error("gpuMapFpgaMem: read: %m");
        gpuUnmapFpgaMem(&b->bwrite);
        return -1;
    }

    return 0;
}

void gpuDestroyBufferState(GpuBufferState_t* b)
{
    gpuUnmapFpgaMem(&b->bwrite);
    gpuUnmapFpgaMem(&b->bread);
}

int gpuMapHostFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size)
{
    memset(outmem, 0, sizeof(*outmem));

    outmem->ptr = (uint8_t*)dmaMapRegister(fd, offset, size);
    if (!outmem->ptr || outmem->ptr == MAP_FAILED) {
        return -1;
    }
    outmem->size = size;
    outmem->fd = fd;

    CUresult status = cuMemHostRegister(outmem->ptr, size, CU_MEMHOSTREGISTER_IOMEMORY);
    if (chkError(status)) {
        logging::error("Unable to map offset=%lu, size=%zu to GPU", offset, size);
        dmaUnMapRegister(fd, (void**)outmem->ptr, outmem->size);
        outmem = {};
        return -1;
    }

    status = cuMemHostGetDevicePointer(&outmem->dptr, outmem->ptr, 0);
    if (chkError(status)) {
        logging::error("Failed to get device pointer: %d", status);
        dmaUnMapRegister(fd, (void**)outmem->ptr, outmem->size);
        outmem = {};
        return -1;
    }

    int flag = 1;
    chkFatal(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, outmem->dptr));

    return 0;
}

int gpuMapFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size, int write)
{
    CUresult result = {};
    cudaError_t err = {};
    memset(outmem, 0, sizeof(*outmem));

    if ((size & 0xFFFF) != 0) {
        logging::error("gpuMapFpgaMem: Size MUST be a multiple of 64k!!");
        return -1;
    }

    uint8_t* dp = 0;
    if ((err = cudaMalloc(&dp, size)) != cudaSuccess) {
        logging::error("cudaMalloc(%zu): %s", size, errorString(err).c_str());
        return -1;
    }
    outmem->dptr = reinterpret_cast<CUdeviceptr>(dp);
    outmem->size = size;
    cudaMemset((void*)outmem->dptr, 0, size);

    int flag = 1;
    // This attribute is required for peer shared memory. It will synchronize every synchronous memory operation on this block of memory.
    if ((result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, outmem->dptr)) != CUDA_SUCCESS) {
        logging::error("cuPointerSetAttribute: %d (%s)", result, errorString(result).c_str());
        cuMemFree(outmem->dptr);
        outmem = {};
        return -1;
    }

    if (gpuAddNvidiaMemory(fd, write, outmem->dptr, outmem->size) < 0) {
        logging::critical("gpuAddNvidiaMemory failed: %m");
        abort();
        cuMemFree(outmem->dptr);
        outmem = {};
        return -1;
    }

    outmem->size = size;
    outmem->gpuOnly = 1;
    outmem->fd = fd;

    return 0;
}

void gpuUnmapFpgaMem(GpuDmaBuffer_t* mem)
{
    if (!mem->gpuOnly) {
        dmaUnMapRegister(mem->fd, &mem->ptr, mem->size);
        mem->ptr = NULL;
        mem->size = 0;
    }
    // FIXME: gpuOnly memory cannot be unmapped?
    chkError(cuMemFree(mem->dptr));
}

DmaTgt_t dmaTgtGet(const DataDev& datadev)
{
    // @todo: This line addresses only lane 0
    const uint64_t reg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_AxisDemuxSelect.offset;
    uint32_t regVal;
    auto rc = dmaReadRegister(datadev.fd(), reg, &regVal);
    if (rc) perror("dmaTgtGet: dmaWriteRegister");

    DmaTgt_t tgt;
    switch (regVal) {
        case TGT_CPU:  tgt = TGT_CPU;  break;
        case TGT_GPU:  tgt = TGT_GPU;  break;
        default:       tgt = TGT_ERR;  break;
    }
    return tgt;
}

void dmaTgtSet(const DataDev& datadev, DmaTgt_t tgt)
{
    // @todo: This line addresses only lane 0
    const uint64_t reg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_AxisDemuxSelect.offset;
    auto rc = dmaWriteRegister(datadev.fd(), reg, tgt);
    if (rc) perror("dmaTgtSet: dmaWriteRegister");
}

/** Function to reset the DMA buffer index */
void dmaIdxReset(const DataDev& datadev)
{
    // Toggle the writeEnable register to reset the DMA buffer index
    const uint64_t reg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_WriteEnable.offset;
    uint32_t value;
    auto rc = dmaReadRegister(datadev.fd(), reg, &value);
    rc = dmaWriteRegister(datadev.fd(), reg, value & ~GpuAsyncReg_WriteEnable.bitMask);
    if (rc) perror("dmaIdxReset: dmaWriteRegister 1");
    rc = dmaWriteRegister(datadev.fd(), reg, value);
    if (rc) perror("dmaIdxReset: dmaWriteRegister 2");
}
