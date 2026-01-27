#include "MemPool.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/range.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/aes-stream-drivers/GpuAsyncRegs.h"

using logging = psalg::SysLog;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


namespace Drp {
  namespace Gpu {

    [[maybe_unused]]
    static const unsigned GPU_OFFSET      = GPU_ASYNC_CORE_OFFSET;
    static const size_t   DMA_BUFFER_SIZE = (sizeof(DmaDsc) + sizeof(TimingHeader) +
                                             6 * 192*168 * sizeof(uint16_t)); // ePixUHR (6 ASICs, 2B/pixel)
  } // Gpu
} // Drp

static void chkMemory(const void* pointer, unsigned count, size_t size, const char* name)
{
  if (!pointer) {
    logging::critical("cudaMalloc returned no memory for %u %s of size %zu\n", count, name, size);
    exit(-ENOMEM);
  }
}

MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool           (para),
  m_setMaskBytesDone(0),
  m_hostWrtBufsSize (0),
  m_hostWrtBufs_d   (nullptr),
  m_calibBufsSize   (0),
  m_calibBuffers_d  (nullptr),
  m_reduceBufsSize  (0),
  m_reduceBuffers_d (nullptr)
{
  // @todo: Get the DMA size from somewhere - query the device?  No, the device is told below.
  //        Get it from the command line?  Get it from the GPU::Detector?
  if (para.kwargs.find("dmaSize") != para.kwargs.end())
    m_dmaSize = std::stoul(const_cast<Parameters&>(para).kwargs["dmaSize"]);
  else
    m_dmaSize = DMA_BUFFER_SIZE;
  m_dmaSize  = ((m_dmaSize + 0xffff) >> 16) << 16; // Round up to multiple of 64 kB for alignment requirement
  m_dmaCount = MAX_BUFFERS;             // @todo: Find this out from the f/w
  dmaBuffers = nullptr;                 // Unused: cause a crash if accessed

  ////////////////////////////////////////////
  // Setup GPU
  ////////////////////////////////////////////

  if (para.verbose)
    m_context.listDevices();

  unsigned gpuId = 0;
  if (para.kwargs.find("gpuId") != para.kwargs.end())
    gpuId = std::stoul(const_cast<Parameters&>(para).kwargs["gpuId"]);

  if (!m_context.init(gpuId, para.verbose == 0)) {
    logging::critical("CUDA initialize failed");
    abort();
  }

  struct RequiredExts_t {
    CUdevice_attribute attr;
    const char* name;
  } RequiredExts[] {
  {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, "GPUDirectRDMA"},
    //{CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, "StreamMemOpsV1"},
  {CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, "MapHostMemory"},
    };

  for (auto e : RequiredExts) {
    if (!m_context.getAttribute(e.attr)) {
      logging::critical("Device is missing required extension: %s", e.name);
      abort();
    }
  }

  logging::info("GPU Device Attributes:");

  logging::info("  Unified addressing: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING));
  logging::info("  Concurrent kernels: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS));
  logging::info("  Compute preemption: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED));
  logging::info("  Can map host memory: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY));
  logging::info("  Number of multiprocessors: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
  logging::info("  Max threads per multiprocessor: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR));
  logging::info("  Max blocks per multiprocessor: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR));
  logging::info("  Max threads per block: %d\n", m_context.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));

  logging::debug("Done with context setup\n");

  ////////////////////////////////////
  // Setup memory
  ////////////////////////////////////

  // Parse the device spec for the list of unit numbers to handle
  std::vector<int> units;
  auto under = para.device.find("_", 0);
  auto base = para.device.substr(0, under+1);
  getRange(para.device.substr(under+1, para.device.length()), units);

  // Check for normal DAQ mode (datadev device) or simulator mode (null device)
  if (para.device != "/dev/null") {
    for (auto unit : units) {
      std::string device(base + std::to_string(unit));
      auto& panel = m_panels.emplace_back(device);
      logging::info("PGP device '%s' opened", device.c_str());

      // Clear out any left-overs from last time
      int res = gpuRemNvidiaMemory(panel.datadev.fd());
      if (res < 0)  logging::error("Error in gpuRemNvidiaMemory");
      logging::debug("Done with gpuRemNvidiaMemory() cleanup");

      ////////////////////////////////////////////////
      // Create write buffers
      ////////////////////////////////////////////////

      // Allocate DMA buffers on the GPU
      // This handles allocating buffers on the device and registering them with the driver.
      for (unsigned i = 0; i < dmaCount(); ++i) {
        if (gpuMapFpgaMem(&panel.dmaBuffers[i], panel.datadev.fd(), 0, dmaSize(), 1) != 0) {
          logging::critical("Failed to allocate DMA buffers of size %zu for %s at number %zd",
                            dmaSize(), device.c_str(), i);
          abort();
        }
      }

#ifndef HOST_REARMS_DMA
      ////////////////////////////////////////////////
      // Map FPGA register space to GPU
      ////////////////////////////////////////////////

      // Map the GpuAsyncCore FPGA registers
      // This causes 'operation not permitted' if the process doesn't have sufficient privileges
      if (gpuMapHostFpgaMem(&panel.swFpgaRegs, panel.datadev.fd(), GPU_OFFSET, 0x100000) < 0) {
        logging::critical("Failed to map GpuAsyncCore at offset=%d, size = %d", GPU_OFFSET, 0x100000);
        logging::info("Consider rebuilding with HOST_REARMS_DMA defined in MemPool.hh");
        abort();
      }

      logging::debug("Mapped FPGA registers");

      // @todo: For now we complain if the number of DMA buffers in f/w doesn't match MAX_BUFFERS.
      //        In the future, we should ensure all units have the same value and use that number.
      auto info1 = readRegister<unsigned>(panel.swFpgaRegs.ptr, GPU_ASYNC_INFO1_REG);
      auto maxBuffers = GPU_ASYNC_INFO1_MAX_BUFFERS(info1);
      if (maxBuffers != MAX_BUFFERS) {
        logging::warning("%s MAX_BUFFERS mismatch: %d vs %d expected",
                         device.c_str(), maxBuffers, MAX_BUFFERS);
        if (maxBuffers < MAX_BUFFERS)  abort();
      }

      for (unsigned i = 0; i < dmaCount(); ++i) {
        auto dmaBufAddr0 = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i));
        auto dmaBufAddr1 = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i)+4);
        auto dmaBufSize  = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_SIZE(i));
        uint64_t dmaBufAddr = ((uint64_t)dmaBufAddr1 << 32) | dmaBufAddr0;
        logging::debug("DMA buffer[%d] dptr %p, addr %p, size %u",
                       i, (void*)panel.dmaBuffers[i].dptr, (void*)dmaBufAddr, dmaBufSize);
      }
#endif

      logging::debug("Done with device mem alloc for %s\n", device.c_str());
    }
  } else {                              // Simulator mode
      auto& panel = m_panels.emplace_back(para.device);
      logging::info("PGP device '%s' opened", para.device.c_str());

      ////////////////////////////////////////////////
      // Create write buffers
      ////////////////////////////////////////////////

      // Allocate "DMA" buffers on the GPU
      for (unsigned i = 0; i < dmaCount(); ++i) {
          uint8_t* dp{nullptr};
          chkError(cudaMalloc(&dp,    dmaSize()));
          chkMemory          ( dp,    dmaSize(), sizeof(*dp), "dmaBuffers");
          chkError(cudaMemset( dp, 0, dmaSize()));
          panel.dmaBuffers[i].ptr     = nullptr;
          panel.dmaBuffers[i].dptr    = reinterpret_cast<CUdeviceptr>(dp);
          panel.dmaBuffers[i].size    = dmaSize();
          panel.dmaBuffers[i].gpuOnly = 1;
          panel.dmaBuffers[i].fd      = 0;
      }

      uint8_t* dp{nullptr};
      uint8_t* hp{nullptr};
      size_t regBlkSize{0x600 * sizeof(uint32_t)};
      chkError(cudaHostAlloc(&hp,    regBlkSize, cudaHostAllocDefault));
      chkMemory             ( hp,    regBlkSize, sizeof(*hp), "swFpgaRegs");
      chkError(cudaMemset   ( hp, 0, regBlkSize));
      chkError(cudaHostGetDevicePointer(&dp, hp, 0));
      panel.swFpgaRegs.ptr     = hp;
      panel.swFpgaRegs.dptr    = reinterpret_cast<CUdeviceptr>(dp);
      panel.swFpgaRegs.size    = regBlkSize;
      panel.swFpgaRegs.gpuOnly = 0;
      panel.swFpgaRegs.fd      = 0;

      // No need to call setMaskBytes, so fake done
      m_setMaskBytesDone = m_panels.size();
  }

  // Initialize the base class before using dependencies like nbuffers()
  _initialize(para);
  pgpEvents.resize(m_nbuffers); // @todo: Revisit this hack - w/o this have only MAX_BUFFERS buffers

  // Set up intermediate buffer pointers
  m_hostWrtBufsVec_h.resize(m_panels.size());
  m_hostWrtBufsVec_d.resize(m_panels.size());
  chkError(cudaMalloc(&m_hostWrtBufs_d,    m_panels.size() * sizeof(*m_hostWrtBufs_d)));
  chkMemory          ( m_hostWrtBufs_d,    m_panels.size(),  sizeof(*m_hostWrtBufs_d), "hostWrtBufs");
  chkError(cudaMemset( m_hostWrtBufs_d, 0, m_panels.size() * sizeof(*m_hostWrtBufs_d)));

  chkError(cudaMalloc(&m_reduceBuffers_d,    nbuffers() * sizeof(*m_reduceBuffers_d)));
  chkMemory          ( m_reduceBuffers_d,    nbuffers(),  sizeof(*m_reduceBuffers_d), "reduceBuffers");
  chkError(cudaMemset( m_reduceBuffers_d, 0, nbuffers() * sizeof(*m_reduceBuffers_d)));

  logging::debug("Done with setting up memory\n");
}

MemPoolGpu::~MemPoolGpu()
{
  printf("*** MemPoolGpu dtor 1\n");
  for (auto& panel : m_panels) {
    // Free the DMA buffers
    for (unsigned i = 0; i < dmaCount(); ++i) {
      gpuUnmapFpgaMem(&panel.dmaBuffers[i]);
    }

    // Release the memory held by the driver
    ssize_t rc;
    if ((rc = gpuRemNvidiaMemory(panel.datadev.fd())) < 0)
      logging::error("gpuRemNvidiaMemory failed: %zd: %M", rc);
  }
  printf("*** MemPoolGpu dtor 2\n");

  // Free the intermediate buffers
  destroyReduceBuffers();
  destroyCalibBuffers();
  for (unsigned i = 0; i < m_panels.size(); ++i) {
    destroyHostBuffers(i);
  }
  printf("*** MemPoolGpu dtor 3\n");

  chkError(cudaFree(m_hostWrtBufs_d));
  printf("*** MemPoolGpu dtor 4\n");

  m_panels.clear();
  printf("*** MemPoolGpu dtor 5\n");
}

int MemPoolGpu::fd(unsigned unit) const
{
  if (unit < m_panels.size())  return m_panels[unit].datadev.fd();

  logging::critical("MemPoolGpu::fd(): unit %u is out of range [0:%zu]",
                    unit, m_panels.size()-1);
  abort();
}

int MemPoolGpu::setMaskBytes(uint8_t laneMask, unsigned virtChan)
{
  int retval = 0;
  if (m_setMaskBytesDone == m_panels.size()) {
    logging::debug("%s: earlier setting in effect", __PRETTY_FUNCTION__);
  } else {
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (unsigned i=0; i<PGP_MAX_LANES; i++) {
      if (laneMask & (1 << i)) {
        uint32_t channel = i;
        uint32_t dest = dmaDest(channel, virtChan);
        logging::info("setting lane  %u, dest 0x%x", i, dest);
        dmaAddMaskBytes(mask, dest);
      }
    }
    for (const auto& panel : m_panels) {
      if (dmaSetMaskBytes(panel.datadev.fd(), mask)) {
        retval = 1; // error
      } else {
        ++m_setMaskBytesDone;
      }
    }
  }
  return retval;
}

void MemPoolGpu::createHostBuffers(unsigned panel, size_t size)
{
  assert(panel < m_hostWrtBufsVec_h.size());

  if (m_hostWrtBufsSize && (size != m_hostWrtBufsSize)) {
    logging::critical("HostWrtBuf size mismatch for panel %u: %zu vs %zu",
                      panel, size, m_hostWrtBufsSize);
    exit(EXIT_FAILURE);
  }

  // Allocate buffers for the DMA descriptors, TimingHeaders and TEB input data
  // in pinned memory and make them visible on both the CPU and the GPU
  auto nBufs = nbuffers();
  chkError(cudaHostAlloc(&m_hostWrtBufsVec_h[panel],    nBufs * size, cudaHostAllocDefault));
  chkMemory             ( m_hostWrtBufsVec_h[panel],    nBufs,  size, "hostWrtBufsVec_h");
  chkError(cudaMemset   ( m_hostWrtBufsVec_h[panel], 0, nBufs * size)); // Avoid reading stale data on re-Configure
  chkError(cudaHostGetDevicePointer(&m_hostWrtBufsVec_d[panel], m_hostWrtBufsVec_h[panel], 0));

  // Insert the dptr to the array of buffers into the panel dptr array
  chkError(cudaMemcpy(&m_hostWrtBufs_d[panel], &m_hostWrtBufsVec_d[panel],
                      sizeof(m_hostWrtBufsVec_d[panel]), cudaMemcpyHostToDevice));

  if (!m_hostWrtBufsSize)  m_hostWrtBufsSize = size;

  auto sz = size / sizeof(*m_hostWrtBufsVec_h[panel]);
  logging::info("Host write buffers for panel %u: %p : %p, size %u * %zu B\n", panel,
                &m_hostWrtBufsVec_h[panel][0], &m_hostWrtBufsVec_h[panel][(nBufs-1) * sz], nBufs, size);
}

void MemPoolGpu::destroyHostBuffers(unsigned panel)
{
  if (m_hostWrtBufsSize) {
    chkError(cudaFreeHost(m_hostWrtBufsVec_h[panel]));
    if (m_hostWrtBufsVec_d[panel]) {
      m_hostWrtBufsVec_d[panel] = nullptr;
    }
    m_hostWrtBufsVec_h[panel] = nullptr;
    m_hostWrtBufsSize = 0;
  }
}

void MemPoolGpu::createCalibBuffers(unsigned nPanels, unsigned nElements)
{
  if (m_calibBufsSize) {
    logging::error("Attempt to reallocate CalibBuffers");
    return;
  }

  // For a each panel, allocate nBufs buffers of nElements for calibrated data on the GPU
  // This space is organized so that for each buffer, the panel data is contiguous, i.e.,
  // calibBuffers[nBufs][nPanels * nElements]
  auto nBufs = nbuffers();
  auto size  = nPanels * nElements * sizeof(*m_calibBuffers_d);
  chkError(cudaMalloc(&m_calibBuffers_d,    nBufs * size));
  chkMemory          ( m_calibBuffers_d,    nBufs,  size, "calibBuffer_d");
  chkError(cudaMemset( m_calibBuffers_d, 0, nBufs * size));

  m_calibBufsSize = size;

  auto sz = size / sizeof(*m_calibBuffers_d);
  logging::info("Calibration buffers: %p : %p, size %u * %zu B\n",
                &m_calibBuffers_d[0], &m_calibBuffers_d[(nBufs-1) * sz], nBufs, size);
}

void MemPoolGpu::destroyCalibBuffers()
{
  if (m_calibBufsSize) {
    chkError(cudaFree(m_calibBuffers_d));
    m_calibBufsSize = 0;
  }
}

void MemPoolGpu::createReduceBuffers(size_t nBytes, size_t reserved)
{
  if (m_reduceBufsSize) {
    logging::error("Attempt to reallocate ReduceBuffers");
    return;
  }

  // Round up both nBytes and reserved to an integer number of uint64_ts for
  // buffer alignment purposes
  nBytes   = sizeof(uint64_t)*((nBytes   + sizeof(uint64_t)-1)/sizeof(uint64_t));
  reserved = sizeof(uint64_t)*((reserved + sizeof(uint64_t)-1)/sizeof(uint64_t));

  // For a each panel, allocate nBufs buffers for reduced data on the GPU,
  // reserving space at the front for the datagram header
  uint8_t* reduceBufferBase;
  auto   nBufs = nbuffers();
  size_t size  = (nBytes + reserved) * sizeof(*m_reduceBuffers_d);
  chkError(cudaMalloc(&reduceBufferBase,    nBufs * size));
  chkMemory          ( reduceBufferBase,    nBufs,  size, "reduceBufferBase");
  chkError(cudaMemset( reduceBufferBase, 0, nBufs * size));
  m_reduceBuffers_d = reduceBufferBase + reserved;

  m_reduceBufsSize = nBytes;           // Doesn't include the reserved portion!
  m_reduceBufsRsvd = reserved;

  auto sz = size / sizeof(*m_reduceBuffers_d);
  logging::info("Reduce buffers: [base %p] %p : %p, size %u * (%zu + %zu) B\n", reduceBufferBase,
                &m_reduceBuffers_d[0], &m_reduceBuffers_d[(nBufs-1) * sz], nBufs, reserved, nBytes);
}

void MemPoolGpu::destroyReduceBuffers()
{
  if (m_reduceBufsSize) {
    chkError(cudaFree(m_reduceBuffers_d - m_reduceBufsRsvd));
    m_reduceBufsSize = 0;
    m_reduceBufsRsvd = 0;
  }
}
