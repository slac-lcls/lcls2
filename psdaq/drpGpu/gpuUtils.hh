#ifndef GPU_UTILS_HH
#define GPU_UTILS_HH

#include "psdaq/aes-stream-drivers/GpuAsync.h"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <string>
#include <memory>
#include <assert.h>
#include <stdint.h>

#if defined(__CUDACC_VER_MAJOR__) || defined(__clangd__)
#define _CUDA
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __NVCC__
#define globalFunc __global__
#define hostFunc __host__
#define deviceFunc __device__
#else
#define deviceFunc
#define globalFunc
#define hostFunc
#endif

namespace Drp {
  namespace Gpu {

//--------------------------------------------------------------------------------//
// CUDA Prototypes
#define NOARG ""           // Ensures there is an arg when __VA_ARGS__ is blank
#define chkFatal(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, true,  NOARG __VA_ARGS__)
#define chkError(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, false, NOARG __VA_ARGS__)

bool checkError(CUresult  status, const char* func, const char* file, int line, bool crash=true, const char* msg="");
bool checkError(cudaError status, const char* func, const char* file, int line, bool crash=true, const char* msg="");
//--------------------------------------------------------------------------------//

/**
 * Wraps a cuda context and handles some initialization for you.
 */
class CudaContext
{
public:
    CudaContext() { chkFatal(cuInit(0)); }

    /**
     * Creates a CUDA context, selects a device and ensures that stream memory ops are available.
     * \param device If >= 0, selects a device to use
     * \param quiet Wheter to spew or not
     * \returns Bool if success
     */
    bool init(int device = -1, bool quiet = false);

    /**
     * \brief Dumps a list of devices to stdout
     */
    void listDevices();

    int getAttribute(CUdevice_attribute attr);

    CUdevice device() const { return device_; }
    CUcontext context() const { return context_; }
    int deviceNo() const { return _devNo; }

    CUcontext context_;
    CUdevice device_;
private:
    int _devNo;
};

//-----------------------------------------------------------------------------//

class CoreRegisters
{
public:
  CoreRegisters() : _swRegs(nullptr), _fwRegs(nullptr) {}
  void initialize(bool sim, void* regs);

  uint32_t dmaDataBytes() const
  { return _fwRegs ? _fwRegs->dmaDataBytes()
                   : _readSwReg(_swRegs, GpuAsyncReg_DmaDataBytes); }
  void     setDataBytes(uint32_t val)
  { if    (_swRegs)  _writeSwReg(_swRegs, GpuAsyncReg_DmaDataBytes, val); }
  uint32_t writeEnable() const
  { return _fwRegs ? _fwRegs->writeEnable()
                   : _readSwReg(_swRegs, GpuAsyncReg_WriteEnableV1); }
  void     setWriteEnable(uint32_t val)
  { if    (_fwRegs)  _fwRegs->setWriteEnable(val);
    else             _writeSwReg(_swRegs, GpuAsyncReg_WriteEnableV1, val); }
  void     setWriteCount(uint32_t val)
  { if    (_fwRegs)  _fwRegs->setWriteCount(val);
    else             _writeSwReg(_swRegs, GpuAsyncReg_WriteCountV1, val); }
  uint32_t axisDeMuxSelect() const
  { return _fwRegs ? _fwRegs->axisDeMuxSelect()
                   : _readSwReg(_swRegs, GpuAsyncReg_AxisDeMuxSelect); }
  void     setAxisDeMuxSelect(uint32_t val)
  { if    (_fwRegs)  _fwRegs->setAxisDeMuxSelect(val);
    else             _writeSwReg(_swRegs, GpuAsyncReg_AxisDeMuxSelect, val); }
  void     returnFreeListIndex(uint32_t buffer)
  { if    (_fwRegs)  _fwRegs->returnFreeListIndex(buffer);
    else             _writeSwReg(_swRegs, freeListOffset(buffer), 1); }
  uint32_t freeListOffset(uint32_t buffer) const
  { return _fwRegs ? _fwRegs->freeListOffset(buffer)
                   : GPU_ASYNC_REG_WRITE_DETECT_OFFSET_V1(buffer); }
  void     setRemoteWriteMaxSize(uint32_t buffer, uint32_t size)
  { if    (_fwRegs)  _fwRegs->setRemoteWriteMaxSize(buffer, size);
    else             _writeSwReg(_swRegs, GPU_ASYNC_REG_WRITE_SIZE_OFFSET_V1(buffer), size); }
private:
  static uint32_t _readSwReg(const void* baseptr, const struct GpuAsyncRegister& reg) {
    uint32_t val = reg.bitMask & *(const uint32_t*)(((const uint8_t*)baseptr) + reg.offset);
    return val >> reg.bitOffset;
  }
  static uint32_t _readSwReg(const void* baseptr, uint32_t offset) {
    return *(uint32_t*)((uint8_t*)baseptr + offset);
  }
  static void     _writeSwReg(void* baseptr, const struct GpuAsyncRegister& reg, uint32_t value) {
    uint32_t* regp = (uint32_t*)(((uint8_t*)baseptr) + reg.offset);
    *regp = (*regp & ~reg.bitMask) | ((value << reg.bitOffset) & reg.bitMask);
  }
  static void     _writeSwReg(void* baseptr, uint32_t offset, uint32_t value) {
    *(uint32_t*)((uint8_t*)baseptr + offset) = value;
  }
private:
  uint32_t*                         _swRegs;
  std::unique_ptr<GpuAsyncCoreRegs> _fwRegs;
};

//-----------------------------------------------------------------------------//

/**
 * Functions to get and set the DMA destination.
 * \param fd The file descriptor of the PGP PCIe device
 * \param mode The enumerated value of the destination
 */
enum DmaTgt_t { TGT_CPU=0x0, TGT_GPU=0x1, TGT_ERR=-1u };
DmaTgt_t dmaTgtGet(CoreRegisters&);
void dmaTgtSet(CoreRegisters&, DmaTgt_t);

/**
 * Function to reset the DMA buffer round-robin index.
 * \param fd The file descriptor of the PGP PCIe device
 */
void dmaIdxReset(CoreRegisters&);

/**
 * Class for timing various things
 */
struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }
  ~GPUTimer() {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(beg, 0); }
  float stop() {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return ms;
  }
};

  } // Gpu
} // Drp

#endif
