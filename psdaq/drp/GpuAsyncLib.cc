#include "GpuAsyncLib.hh"

#include "psdaq/aes-stream-drivers/GpuAsyncRegs.h"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;


bool checkError(CUresult status, const char* func, const char* file, int line, bool crash, const char* msg)
{
  if (status != CUDA_SUCCESS) {
    const char* perrstr = 0;
    CUresult    rc      = cuGetErrorString(status, &perrstr);
    if (rc == CUDA_SUCCESS) {
      if (perrstr) {
        logging::error("%s:%d:\n  '%s'\n  status %d: info: %s - %s\n", file, line, func, status, perrstr, msg);
      } else {
        logging::error("%s:%d:\n  '%s'\n  status %d: info: unknown error - %s\n", file, line, func, status, msg);
      }
    } else {
      logging::error("%s:%d:\n  '%s'\n  status %d: info: unknown error - %s\n", file, line, func, status, msg);
    }
    if (crash)  abort();
    return true;
  }
  return false;
}

bool checkError(cudaError status, const char* func, const char* file, int line, bool crash, const char* msg)
{
  if (status != cudaSuccess) {
    logging::error("%s:%d:  '%s'\n  %s\n  status %d: info: %s - %s\n", file, line, func, status, cudaGetErrorString(status), msg);
    if (crash)  abort();
    return true;
  }
  return false;
}

// -------------------------------------------------------------------

DataGPU::DataGPU(const char* path) {
    m_fd = open(path, O_RDWR);
    if (m_fd < 0) {
        logging::critical("Error opening %s: %m", path);
        abort();
    }
}

// -------------------------------------------------------------------

CudaContext::CudaContext()
{
  chkFatal(cuInit(0), "Error while initting cuda");
}

bool CudaContext::init(int device)
{
  int devs = 0;
  if (chkError(cuDeviceGetCount(&devs)))
    return false;
  logging::debug("Total GPU devices %d\n", devs);
  if (devs <= 0) {
    logging::error("No GPU devices available!\n");
    return false;
  }

  device = device < 0 ? 0 : device;
  if (devs <= device) {
    logging::error("Invalid GPU device number %d! There are only %d devices available\n", device, devs);
    return false;
  }

  // Actually get the device...
  if (chkError(cuDeviceGet(&m_device, device), "Could not get GPU device!"))
    return false;

  // Spew device name
  char name[256];
  if (chkError(cuDeviceGetName(name, sizeof(name), m_device)))
    return false;
  logging::debug("Selected GPU device: %s\n", name);

  // Set required attributes
  int res;
  if (chkError(cuDeviceGetAttribute(&res, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, m_device)))
    return false;
  if (!res) {
    logging::warning("This device does not support CUDA Stream Operations, this code will not run!\n");
    logging::error("  Consider setting NVreg_EnableStreamMemOPs=1 when loading the NVIDIA kernel module, "
                   "if your GPU is supported.\n");
    return false;
  }

  // Report memory totals
  size_t global_mem = 0;
  if (chkError(cuDeviceTotalMem(&global_mem, m_device)))
    return false;
  logging::debug("Global memory: %zu MB\n", global_mem >> 20);
  if (global_mem > (size_t)4 << 30)
    logging::debug("64-bit Memory Address support\n");

  int value;
  if (chkError(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, m_device)))
    return false;
  logging::debug("Device supports unified addressing: %s\n", value ? "YES" : "NO");

  // Create context
  if (chkError(cuCtxCreate(&m_context, 0, m_device)))
    return false;

  return true;
}

void CudaContext::listDevices()
{
  int devs = 0;
  if (chkError(cuDeviceGetCount(&devs), "Unable to get device count"))
    return;

  for (int i = 0; i < devs; ++i) {
    CUdevice dev;
    if (chkError(cuDeviceGet(&dev, i))) {
      logging::error("Unable to get device %d", i);
      continue;
    }
    char name[256];
    if (chkError(cuDeviceGetName(name, sizeof(name), dev)))
      break;
    logging::info("%d: %s\n", i, name);
  }
}

//-----------------------------------------------------------------------------

int gpuInitBufferState(GpuBufferState_t* b, const DataGPU& gpu, size_t bufSize)
{
    // Allocate buffers on the GPU
    if (gpuMapFpgaMem(&b->bwrite, gpu.fd(), 0, bufSize, 1) != 0) {
        perror("gpuMapFpgaMem: write");
        return -1;
    }

    if (gpuMapFpgaMem(&b->bread, gpu.fd(), 0, bufSize, 0) != 0) {
        perror("gpuMapFpgaMem: read");
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
        logging::error("Unable to map offset=%lu, size=%zu to GPU\n", offset, size);
        dmaUnMapRegister(fd, (void**)outmem->ptr, outmem->size);
        outmem = {};
        return -1;
    }

    status = cuMemHostGetDevicePointer(&outmem->dptr, outmem->ptr, 0);
    if (chkError(status)) {
        logging::error("Failed to get device pointer: %d\n", status);
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
    memset(outmem, 0, sizeof(*outmem));

    if (cuMemAlloc(&outmem->dptr, size) != CUDA_SUCCESS) {
        return -1;
    }
    outmem->size = size;
    cuMemsetD8(outmem->dptr, 0, size);

    int flag = 1;
    // This attribute is required for peer shared memory. It will synchronize every synchronous memory operation on this block of memory.
    if (cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, outmem->dptr) != CUDA_SUCCESS) {
        cuMemFree(outmem->dptr);
        outmem = {};
        return -1;
    }

    if (gpuAddNvidiaMemory(fd, write, outmem->dptr, outmem->size) < 0) {
        logging::critical("gpuAddNvidiaMemory failed\n");
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

DmaTgt_t dmaTgtGet(const DataGPU& gpu)
{
    // @todo: This line addresses only lane 0
    const uint64_t dynRtReg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_DynamicRouteMasks0.offset;
    uint32_t regVal;
    auto rc = dmaReadRegister(gpu.fd(), dynRtReg, &regVal);
    if (rc) perror("dmaTgtGet: dmaWriteRegister");

    DmaTgt_t tgt;
    switch (regVal) {
        case CPU:  tgt = CPU;  break;
        case GPU:  tgt = GPU;  break;
        default:   tgt = ERR;  break;
    }
    return tgt;
}

void dmaTgtSet(const DataGPU& gpu, DmaTgt_t tgt)
{
    // @todo: This line addresses only lane 0
    const uint64_t dynRtReg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_DynamicRouteMasks0.offset;
    auto rc = dmaWriteRegister(gpu.fd(), dynRtReg, tgt);
    if (rc) perror("dmaTgtSet: dmaWriteRegister");
}

void dmaIdxReset(const DataGPU& gpu)
{
    const uint64_t writeEnReg = GPU_ASYNC_CORE_OFFSET + GpuAsyncReg_WriteEnable.offset;
    uint32_t value;
    auto rc = dmaReadRegister(gpu.fd(), writeEnReg, &value);
    rc = dmaWriteRegister(gpu.fd(), writeEnReg, value & ~GpuAsyncReg_WriteEnable.bitMask);
    if (rc) perror("dmaIdxReset: dmaWriteRegister 1");
    rc = dmaWriteRegister(gpu.fd(), writeEnReg, value);
    if (rc) perror("dmaIdxReset: dmaWriteRegister 2");
}
