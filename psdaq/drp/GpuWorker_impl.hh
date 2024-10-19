#pragma once

#include <cstddef>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "GpuWorker.hh"


namespace Drp {

class Parameters;
class PGPEvent;
class Pebble;
class MemPool;

struct Batch
{
  uint32_t start;
  uint32_t size;
};

/**
 * Wraps common CUDA init code and whatnot.
 */
class CudaContext
{
public:
    CudaContext();

    /**
     * Creates a CUDA context, selects a device and ensures that stream memory ops are available.
     * \param device If >= 0, selects a device to use
     * \param quiet Whether to spew or not
     * \returns Bool if success
     */
    bool initialize(int device = -1);

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

class GpuMemPool
{
public:
  GpuMemPool(const Parameters& para, MemPool& pool);
  ~GpuMemPool();

  unsigned  count()    const;
  size_t    dmaSize()  const;
  unsigned  nbuffers() const;
  int       fd()       const;
  Pebble&   pebble()   const;
  std::vector<PGPEvent>& pgpEvents() const;
  unsigned  allocate();
  int       initialize();
private:
  /**
   * \brief Maps FPGA memory to the host and allows GPU access to it
   * This is the "standard" way of doing DMA with CUDA. cuMemHostRegister
   * essentially just gives CUDA access to some pages and allows the GPU to
   * read/write to them.
   * Unlike RDMA, this doesn't require any custom code on the kernel side, this
   * will work with any (properly aligned) buffer given, whether it's IO memory
   * or not. Again unlike RDMA, cuMemHostRegister has more latency and lower
   * throughput, since a DMA transfer must occur between the
   * GPU <-> HOST <-> FPGA
   * \param buffer
   * \param offset Offset of the register block
   * \param size Size of the register block
   * \returns 0 for success
   */
  int _gpuMapHostFpgaMem(GpuDmaBuffer_t& buffer, uint64_t offset, size_t size);

  /**
   * \brief Unmaps memory, clears out the pointer and size
   */
  void _gpuUnmapHostFpgaMem(GpuDmaBuffer_t& buffer);

  /**
   * \brief Maps GPU memory to the FPGA using RDMA
   * This function uses gpuAddNvidiaMemory to give the FPGA access to some pages
   * of memory located on the GPU. This is the other method of doing DMA, and it
   * requires custom driver code and a very specific PCIe topology.
   * Regardless, RDMA allows for low-latency, high-bandwidth DMA transfers
   * between two devices on the bus, without needing to interact with the CPU.
   * \param buffer
   * \param offset Offset of the register block
   * \param size Size of the register block
   * \param write If 1, this will be writable
   */
  int _gpuMapFpgaMem(CUdeviceptr& buffer, uint64_t offset, size_t size, int write);

  /**
   * \brief Unmaps memory, clears out the pointer and size
   */
  void _gpuUnmapFpgaMem(CUdeviceptr& buffer);

public:
  /** Max buffers, must match firmware value **/
  static const unsigned MAX_BUFFERS = 4;
public:
  std::vector<CUdeviceptr> dmaBuffers;
  GpuDmaBuffer_t           swFpgaRegs;
private:
  MemPool& m_pool;
};

class GpuWorker_impl : public GpuWorker
{
public:
  GpuWorker_impl(const Parameters& para, MemPool& pool, Detector& det);
  virtual Detector* detector() override { return &m_det; }
  virtual void timingHeaders(unsigned index, Pds::TimingHeader* buffer) override;
  virtual void process(Batch& batch, bool& sawDisable) override;
  virtual void reader(uint32_t start, SPSCQueue<Batch>& collectorGpuQueue) override;
  virtual unsigned lastEvtCtr() const override { return m_lastEvtCtr; }
public:
  uint64_t dmaBytes() const { return m_dmaBytes; }
  uint64_t dmaSize()  const { return m_dmaSize; }
private:
  Detector&             m_det;
  GpuMemPool            m_pool;
  CudaContext           m_context;
  std::vector<CUstream> m_streams;
  unsigned              m_dmaIndex;
  uint32_t              m_lastEvtCtr;
  const Parameters&     m_para;
  uint64_t              m_dmaBytes;
  uint64_t              m_dmaSize;
};

};
