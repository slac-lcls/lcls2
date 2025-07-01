#include "MemPool.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/range.hh"
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/aes-stream-drivers/GpuAsyncRegs.h"

using logging = psalg::SysLog;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


namespace Drp {
  namespace Gpu {

    static const unsigned GPU_OFFSET      = GPU_ASYNC_CORE_OFFSET;
    static const size_t   DMA_BUFFER_SIZE = 384*1024; // ePixUHR = 192*168*6 ASICs * 2B

  } // Gpu
} // Drp

static bool chkMemory(const void* pointer, unsigned count, size_t size, const char* name)
{
  if (!pointer) {
    logging::error("cudaMalloc returned no memory for %u %s of size %zu\n", count, name, size);
    return true;
  }
  return false;
}

MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool           (para),
  m_setMaskBytesDone(0),
  m_hostWrtBufsSize (0),
  m_hostWrtBufs_d   (nullptr),
  m_calibBufSize    (0),
  m_calibBuffers_d  (nullptr),
  m_reduceBufSize   (0),
  m_reduceBuffers_d (nullptr)
{
  // @todo: Get the DMA size from somewhere - query the device?  No, the device is told below.
  //        Get it from the command line?
  m_dmaSize  = DMA_BUFFER_SIZE;
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
  logging::debug("Done with context setup\n");

  ////////////////////////////////////
  // Setup memory
  ////////////////////////////////////

  std::vector<int> units;
  auto pos = para.device.find("_", 0);
  getRange(para.device.substr(pos+1, para.device.length()), units);
  for (auto unit : units) {
    std::string device(para.device.substr(0, pos+1) + std::to_string(unit));
    auto& panel = m_panels.emplace_back(device);
    logging::info("PGP device '%s' opened", device.c_str());

    // Clear out any left-overs from last time
    int res = gpuRemNvidiaMemory(panel.gpu.fd());
    if (res < 0)  logging::error("Error in gpuRemNvidiaMemory\n");
    logging::debug("Done with gpuRemNvidiaMemory() cleanup\n");

    ////////////////////////////////////////////////
    // Map FPGA register space to GPU
    ////////////////////////////////////////////////

    /** Map the GpuAsyncCore FPGA registers **/
    if (gpuMapHostFpgaMem(&panel.swFpgaRegs, panel.gpu.fd(), GPU_OFFSET, 0x100000) < 0) {
      logging::critical("Failed to map GpuAsyncCore at offset=%d, size = %d", GPU_OFFSET, 0x100000);
      abort();
    }

    /** Compute 'write start' register location using the device pointer to GpuAsyncCore **/
    panel.hwWriteStart = panel.swFpgaRegs.dptr + 0x300;

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

    // Allocate DMA buffers on the GPU
    // This handles allocating buffers on the device and registering them with the driver.
    for (unsigned i = 0; i < dmaCount(); ++i) {
      if (gpuMapFpgaMem(&panel.dmaBuffers[i], panel.gpu.fd(), 0, dmaSize(), 1) != 0) {
        logging::critical("Failed to alloc buffer list for %s at number %zd", device.c_str(), i);
        abort();
      }
      auto dmaBufAddr0 = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i));
      auto dmaBufAddr1 = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i)+4);
      auto dmaBufSize  = readRegister<uint32_t>(panel.swFpgaRegs.ptr, GPU_ASYNC_WR_SIZE(i));
      uint64_t dmaBufAddr = ((uint64_t)dmaBufAddr1 << 32) | dmaBufAddr0;
      printf("*** DMA buffer[%d] dptr %p, addr %p, size %u\n",
             i, (void*)panel.dmaBuffers[i].dptr, (void*)dmaBufAddr, dmaBufSize);
    }

    logging::debug("Done with device mem alloc for %s\n", device.c_str());
  }

  // Initialize the base class before using dependencies like nbuffers()
  _initialize(para);

  // Set up intermediate buffer pointers
  m_hostWrtBufsVec_h.resize(m_panels.size());
  m_hostWrtBufsVec_d.resize(m_panels.size());
  chkError(cudaMalloc(&m_hostWrtBufs_d,    m_panels.size() * sizeof(*m_hostWrtBufs_d)));
  chkMemory          ( m_hostWrtBufs_d,    m_panels.size(),  sizeof(*m_hostWrtBufs_d), "hostWrtBufs");
  chkError(cudaMemset( m_hostWrtBufs_d, 0, m_panels.size() * sizeof(*m_hostWrtBufs_d)));
  printf("*** MemPool ctor: hostWrtBufs_d %p\n", m_hostWrtBufs_d);

  m_calibBufsVec_h.resize(nbuffers());
  chkError(cudaMalloc(&m_calibBuffers_d,    nbuffers() * sizeof(*m_calibBuffers_d)));
  chkMemory          ( m_calibBuffers_d,    nbuffers(),  sizeof(*m_calibBuffers_d), "calibBuffers");
  chkError(cudaMemset( m_calibBuffers_d, 0, nbuffers() * sizeof(*m_calibBuffers_d)));
  printf("*** MemPool ctor: calibBuffers_d %p\n", m_calibBuffers_d);

  m_reduceBufsVec_h.resize(nbuffers());
  chkError(cudaMalloc(&m_reduceBuffers_d,    nbuffers() * sizeof(*m_reduceBuffers_d)));
  chkMemory          ( m_reduceBuffers_d,    nbuffers(),  sizeof(*m_reduceBuffers_d), "reduceBuffers");
  chkError(cudaMemset( m_reduceBuffers_d, 0, nbuffers() * sizeof(*m_reduceBuffers_d)));
  printf("*** MemPool ctor: reduceBuffers_d %p\n", m_reduceBuffers_d);

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
    if ((rc = gpuRemNvidiaMemory(panel.gpu.fd())) < 0)
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

int MemPoolGpu::fd() const
{
  // @todo: Need to know for which device this is being called
  logging::critical("MemPoolGpu::fd() called: Unsupported");
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
      if (dmaSetMaskBytes(panel.gpu.fd(), mask)) {
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

  // Allocate buffers for the DMA descriptors, TimingHeaders and TEB input data
  // in managed memory so that they are visible on both the CPU and the GPU
  auto nBufs = nbuffers();
  printf("*** createHostBufs 1: sz %zu, nbufs %u\n", m_hostWrtBufsVec_h.size(), nBufs);
  m_hostWrtBufsVec_h[panel].resize(nBufs);
  printf("*** createHostBufs 2: sz %zu\n", m_hostWrtBufsVec_h[panel].size());
  // Allocate a device array to store nBufs ptrs
  chkError(cudaMalloc(&m_hostWrtBufsVec_d[panel], nBufs * sizeof(*m_hostWrtBufsVec_d[panel])));
  chkMemory          ( m_hostWrtBufsVec_d[panel], nBufs,  sizeof(*m_hostWrtBufsVec_d[panel]), "hostWrtBufsVec");
  printf("*** createHostBufs 2a: hostWrtBufsVec_d %p\n", m_hostWrtBufsVec_d[panel]);
  // Allocate managed memory for each of the nBufs buffers and store the ptr in a host std::vector
  uint8_t* hostWriteBufBase;
  chkError(cudaMallocManaged(&hostWriteBufBase,    nBufs * size));
  chkMemory                 ( hostWriteBufBase,    nBufs,  size, "hostWriteBuf");
  chkError(cudaMemset       ( hostWriteBufBase, 0, nBufs * size)); // Avoid rereading junk on re-Configure
  for (unsigned i = 0; i < nBufs; ++i) {
    m_hostWrtBufsVec_h[panel][i] = (uint32_t*)(hostWriteBufBase + i * size);
  }
  printf("*** createHostBufs 2b: hostWrtBufsVec_h[%u][0] %p\n", panel, m_hostWrtBufsVec_h[panel][0]);
  // Copy the array of managed memory buffer ptrs to the device
  chkError(cudaMemcpy(m_hostWrtBufsVec_d[panel], m_hostWrtBufsVec_h[panel].data(),
                      m_hostWrtBufsVec_h[panel].size() * sizeof(m_hostWrtBufsVec_h[panel][0]),
                      cudaMemcpyHostToDevice));
  printf("*** createHostBufs 2c: hostWrtBufsVec_d[%u] %p\n", panel, m_hostWrtBufsVec_d[panel]);
  // Insert the dptr to the array of buffers into the panel dptr array
  chkError(cudaMemcpy(&m_hostWrtBufs_d[panel], &m_hostWrtBufsVec_d[panel],
                      sizeof(m_hostWrtBufsVec_d[panel]), cudaMemcpyHostToDevice));
  printf("*** createHostBufs 2d: hostWrtBufs_d %p\n", m_hostWrtBufs_d);

  if (m_hostWrtBufsSize) {
    if (m_hostWrtBufsSize != size) {
      logging::error("HostWrtBuf size mismatch for panel %u: %zu vs %zu",
                     panel, size, m_hostWrtBufsSize);
    }
  } else {
    m_hostWrtBufsSize = size;
  }
  logging::info("Host write buffers for panel %u: %p : %p, sz %zu * %u\n", panel,
                m_hostWrtBufsVec_h[panel][0], m_hostWrtBufsVec_h[panel][nBufs-1], size, nBufs);
  printf("*** createHostBufs 3: size %zu\n", size);
}

void MemPoolGpu::destroyHostBuffers(unsigned panel)
{
  if (m_hostWrtBufsSize) {
    chkError(cudaFree(m_hostWrtBufsVec_h[panel][0]));
    if (m_hostWrtBufsVec_d[panel]) {
      chkError(cudaFree(m_hostWrtBufsVec_d[panel]));
      m_hostWrtBufsVec_d[panel] = nullptr;
    }
    m_hostWrtBufsVec_h[panel].clear();
    m_hostWrtBufsSize = 0;
  }
}

void MemPoolGpu::createCalibBuffers(unsigned nPanels, unsigned nElements)
{
  if (m_calibBufSize) {
    logging::error("Attempt to reallocate CalibBuffers");
    return;
  }

  // For a each panel, allocate nBufs buffers of nElements for calibrated data on the GPU
  // This space is organized so that for each buffer, the panel data is contiguous, i.e.,
  // calibBuffers[nBufs][nPanels * nElements]
  auto nBufs = nbuffers();
  m_calibBufsVec_h.resize(nBufs);
  auto size = nPanels * nElements * sizeof(**m_calibBuffers_d);
  float* calibBufferBase;
  chkError(cudaMalloc(&calibBufferBase,    nBufs * size));
  chkMemory          ( calibBufferBase,    nBufs,  size, "calibBuffer");
  chkError(cudaMemset( calibBufferBase, 0, nBufs * size));
  for (unsigned i = 0; i < nBufs; ++i) {
    m_calibBufsVec_h[i] = (float*)((uint8_t*)calibBufferBase + i * size);
  }
  // Copy the array of calibration buffer ptrs to the device
  chkError(cudaMemcpy(m_calibBuffers_d, m_calibBufsVec_h.data(),
                      m_calibBufsVec_h.size() * sizeof(*m_calibBuffers_d),
                      cudaMemcpyHostToDevice));
  m_calibBufSize = size;
  logging::info("Calibration buffers: %p : %p, sz %zu * %u\n",
                m_calibBufsVec_h[0], m_calibBufsVec_h[nBufs-1], size, nBufs);
}

void MemPoolGpu::destroyCalibBuffers()
{
  if (m_calibBufSize) {
    chkError(cudaFree(m_calibBufsVec_h[0]));
    for (auto& calibBuffer : m_calibBufsVec_h) {
      calibBuffer = nullptr;
    }
    m_calibBufsVec_h.clear();
    m_calibBufSize = 0;
  }
}

void MemPoolGpu::createReduceBuffers(size_t nBytes, size_t reserved)
{
  if (m_reduceBufSize) {
    logging::error("Attempt to reallocate ReduceBuffers");
    return;
  }

  // Allocate buffers for reduced data on the GPU,
  // reserving space at the front for the datagram header
  auto nBufs = nbuffers();
  m_reduceBufsVec_h.resize(nBufs);
  uint8_t* reduceBufferBase;
  size_t size = nBytes + reserved;
  chkError(cudaMalloc(&reduceBufferBase,    nBufs * size));
  chkMemory          ( reduceBufferBase,    nBufs,  size, "reduceBuffer");
  chkError(cudaMemset( reduceBufferBase, 0, nBufs * size));
  for (unsigned i = 0; i < nBufs; ++i) {
    m_reduceBufsVec_h[i] = reduceBufferBase + reserved + i * size;
  }
  // Copy the array of data buffer ptrs to the device
  chkError(cudaMemcpy(m_reduceBuffers_d, m_reduceBufsVec_h.data(),
                      m_reduceBufsVec_h.size() * sizeof(*m_reduceBuffers_d),
                      cudaMemcpyHostToDevice));
  m_reduceBufSize = nBytes;             // Doesn't include the reserved portion!
  m_reduceBufRsvd = reserved;
  logging::info("Reduce buffers: [base %p] %p : %p, sz (%zu + %zu) * %u\n", reduceBufferBase,
                m_reduceBufsVec_h[0], m_reduceBufsVec_h[nBufs-1], nBytes, reserved, nBufs);
}

void MemPoolGpu::destroyReduceBuffers()
{
  if (m_reduceBufSize) {
    chkError(cudaFree(m_reduceBufsVec_h[0] - m_reduceBufRsvd));
    for (auto& reduceBuffer : m_reduceBufsVec_h) {
      reduceBuffer = nullptr;
    }
    m_reduceBufsVec_h.clear();
    m_reduceBufSize = 0;
    m_reduceBufRsvd = 0;
  }
}
