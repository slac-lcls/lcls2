/**
 * ----------------------------------------------------------------------------
 * Company    : SLAC National Accelerator Laboratory
 * ----------------------------------------------------------------------------
 * Description: Library and utilities for CUDA and the GPU Async protocol.
 * ----------------------------------------------------------------------------
 * This file is part of 'axi-pcie-devel'. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of 'axi-pcie-devel', including this file, may be copied, modified,
 * propagated, or distributed except according to the terms contained in
 * the LICENSE.txt file.
 * ----------------------------------------------------------------------------
 **/
#pragma once

#include "gpuUtils.h"

#include <string>
#include "psdaq/aes-stream-drivers/GpuAsync.h"
#include <assert.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

//--------------------------------------------------------------------------------//
// CUDA Prototypes
#define NOARG ""           // Ensures there is an arg when __VA_ARGS__ is blank
#define chkFatal(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, true,  NOARG __VA_ARGS__)
#define chkError(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, false, NOARG __VA_ARGS__)

bool checkError(CUresult  status, const char* func, const char* file, int line, bool crash=true, const char* msg="");
bool checkError(cudaError status, const char* func, const char* file, int line, bool crash=true, const char* msg="");
//--------------------------------------------------------------------------------//

/**
 * Wraps a data_gpu device so it can be automatically freed
 */
class DataGPU
{
public:
    DataGPU(const char* path);
    ~DataGPU()
    {
        close(m_fd);
    }

    int fd() const { return m_fd; }

protected:
    int m_fd;
};

/**
 * Wraps a cuda context and handles some initialization for you.
 */
class CudaContext
{
public:
    CudaContext();

    /**
     * Creates a CUDA context, selects a device and ensures that stream memory ops are available.
     * \param device If >= 0, selects a device to use
     * \param quiet Wheter to spew or not
     * \returns Bool if success
     */
    bool init(int device = -1);

    /**
     * \brief Dumps a list of devices
     */
    static void listDevices();

    CUcontext context() const { return m_context; }
    CUdevice  device()  const { return m_device; }

    CUcontext m_context;
    CUdevice  m_device;
};

/**
 * Dumps a buffer from the GPU to a stream
 * \param devicePtr Pointer to the block of memory to dump on the GPU
 * \param sizeInBytes
 * \param buf The copy buffer (assumed to be sizeInBytes in size). If NULL, one will be allocated
 * \param stream The stream to dump to
 */
template<typename T, int COLS = 16>
static inline void dumpGpuMem(CUdeviceptr devicePtr, size_t sizeInBytes, void* buf = nullptr, FILE* stream = stdout)
{
    bool buffed = !buf;
    if (buffed) buf = malloc(sizeInBytes);
    cuMemcpyDtoH(buf, devicePtr, sizeInBytes);
    dumpMem<T, COLS>(reinterpret_cast<T*>(buf), sizeInBytes);
    if (buffed) free(buf);
}

/**
 * \brief Describes FPGA memory that's mapped via RDMA to the GPU. The below functions wrap
 * the creation and destruction process of the memory.
 */
struct GpuDmaBuffer_t
{
  int         fd;
  uint8_t*    ptr;     /** Host accessible pointer **/
  size_t      size;    /** Size of the block **/
  CUdeviceptr dptr;    /** Pointer on the device **/
  int         gpuOnly; /** 1 if this is FPGA <-> GPU only, not mapped to host at all **/
};

/**
 * \brief Maps FPGA memory to the host and allows GPU access to it
 * This is the "standard" way of doing DMA with CUDA. cuMemHostRegister essentially just gives CUDA
 * access to some pages and allows the GPU to read/write to them.
 * Unlike RDMA, this doesn't require any custom code on the kernel side, this will work with any (properly aligned)
 * buffer given, whether it's IO memory or not. Again unlike RDMA, cuMemHostRegister has more latency
 * and lower throughput, since a DMA transfer must occur between the GPU <-> HOST <-> FPGA
 * \param outmem
 * \param fd File descriptor of the FPGA device
 * \param offset Offset of the register block
 * \param size Size of the register block
 * \returns 0 for success
 */
int gpuMapHostFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size);

/**
 * \brief Maps GPU memory to the FPGA using RDMA
 * This function uses gpuAddNvidiaMemory to give the FPGA access to some pages of memory located
 * on the GPU. This is the other method of doing DMA, and it requires custom driver code and a
 * very specific PCIe topology.
 * Regardless, RDMA allows for low-latency, high-bandwidth DMA transfers between two devices on
 * the bus, without needing to interact with the CPU.
 * \param outmem
 * \param fd File descriptor of the FPGA device
 * \param offset Offset of the register block
 * \param size Size of the register block
 * \param write If 1, this will be writable
 */
int gpuMapFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size, int write);

/**
 * \brief Unmaps memory, clears out the pointer and size
 */
void gpuUnMapFpgaMem(GpuDmaBuffer_t* mem);

//-----------------------------------------------------------------------------//

struct GpuBufferState_t
{
    uint8_t* swFpgaRegs;

    GpuDmaBuffer_t bread;
    GpuDmaBuffer_t bwrite;
};

/**
 * Allocates and inits buffer pairs on the GPU for RDMA operation.
 * \param b Buffer state to init
 * \param gpu The DataGPU instance to allocate for
 * \param bufSize The size of the buffers to allocate
 * \return 0 for success, -1 on error
 */
int gpuInitBufferState(GpuBufferState_t* b, const DataGPU& gpu, size_t bufSize);
void gpuDestroyBufferState(GpuBufferState_t* b);

//-----------------------------------------------------------------------------//

/**
 * Functions to get and set the DMA destination.
 * \param fd The file descriptor of the datagpu device
 * \param mode The enumerated value of the destination
 */
enum DmaTgt_t { CPU=0x0000ffff, GPU=0xffff0000, ERR=-1u };
DmaTgt_t dmaTgtGet(int fd);
void dmaTgtSet(int fd, DmaTgt_t);
