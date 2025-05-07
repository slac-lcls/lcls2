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
    static const size_t   DMA_BUFFER_SIZE = 256*1024; // ePixUHR = 192*168*4 ASICs * 2B

  } // Gpu
} // Drp


MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool           (para),
  m_setMaskBytesDone(0),
  m_hostPnlWrBufs_d (nullptr),
  m_calibBuffers    (nullptr),
  m_dataBuffers     (nullptr)
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
  para.nworkers = 0;
  for (auto unit : units) {
    std::string device(para.device.substr(0, pos+1) + std::to_string(unit));
    auto& panel = m_panels.emplace_back(device);
    logging::info("PGP device '%s' opened", device.c_str());

    // There is one worker per PGP device
    para.nworkers++;

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

  // Set up intermediate buffer pointers
  m_hostWriteBufs_h.resize(para.nworkers); // aka, nPanels
  m_hostWriteBufs_d.resize(para.nworkers);
  chkError(cudaMalloc(&m_hostPnlWrBufs_d,    para.nworkers * sizeof(*m_hostPnlWrBufs_d)));
  chkError(cudaMemset( m_hostPnlWrBufs_d, 0, para.nworkers * sizeof(*m_hostPnlWrBufs_d)));
  printf("*** MemPool ctor: hostPnlWrBufs_d %p\n", m_hostPnlWrBufs_d);

  logging::debug("Done with setting up memory\n");

  // Continue with initialization of the base class
  _initialize(para);
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

  chkError(cudaFree(m_hostPnlWrBufs_d));
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

void MemPoolGpu::createHostBuffers(unsigned panel, unsigned nBuffers, size_t size)
{
  assert(panel < m_hostWriteBufs_h.size());

  // Allocate buffers for the DMA descriptors, TimingHeaders and TEB input data
  // in managed memory so that they are visible on both the CPU and the GPU
  printf("*** createHostBufs 1: sz %zu, nbufs %u vs %u\n", m_hostWriteBufs_h.size(), nBuffers, nbuffers());
  m_hostWriteBufs_h[panel].resize(nBuffers);
  printf("*** createHostBufs 2: sz %zu\n", m_hostWriteBufs_h[panel].size());
  // @todo: MallocManaged the entire block of nBuffers * size bytes and then divie it up?
  // @todo: Switch to using pointer math instead of storing nBuffers pointers?
  // Allocate a device array to store nBuffers ptrs
  chkError(cudaMalloc(&m_hostWriteBufs_d[panel], nBuffers * sizeof(*m_hostWriteBufs_d[panel])));
  printf("*** createHostBufs 2a: hostWriteBufs_d %p\n", m_hostWriteBufs_d[panel]);
  // Allocate managed memory for each of the nBuffers buffers and store the ptr in a host std::vector
  for (auto& hostWriteBuf : m_hostWriteBufs_h[panel]) {
    chkError(cudaMallocManaged(&hostWriteBuf,    size)); // @todo: round up size to some alignment value?
    chkError(cudaMemset       ( hostWriteBuf, 0, size)); // Avoid rereading junk on re-Configure
  }
  printf("*** createHostBufs 2b: hostWriteBufs_h[%u][0] %p\n", panel, m_hostWriteBufs_h[panel][0]);
  // Copy the array of managed memory buffer ptrs to the device
  chkError(cudaMemcpy(m_hostWriteBufs_d[panel], m_hostWriteBufs_h[panel].data(),
                      m_hostWriteBufs_h[panel].size() * sizeof(m_hostWriteBufs_h[panel][0]),
                      cudaMemcpyHostToDevice));
  printf("*** createHostBufs 2c: hostWriteBufs_d[%u] %p\n", panel, m_hostWriteBufs_d[panel]);
  // Insert the dptr to the array of buffers into the panel dptr array
  chkError(cudaMemcpy(&m_hostPnlWrBufs_d[panel], &m_hostWriteBufs_d[panel],
                      sizeof(m_hostWriteBufs_d[panel]), cudaMemcpyHostToDevice));
  printf("*** createHostBufs 2d: hostPnlWrBufs_d %p\n", m_hostPnlWrBufs_d);

  printf("*** createHostBufs 3\n");
}

void MemPoolGpu::destroyHostBuffers(unsigned panel)
{
  if (m_hostWriteBufs_d[panel]) {
    for (auto& hostWriteBuf : m_hostWriteBufs_h[panel]) {
      chkError(cudaFree(hostWriteBuf));
    }
    chkError(cudaFree(m_hostWriteBufs_d[panel]));
    m_hostWriteBufs_d[panel] = 0;
  }
}

void MemPoolGpu::createCalibBuffers(unsigned nBuffers, unsigned nPanels, unsigned nWords)
{
  assert(panel < m_hostWriteBufs_h.size());

  // For a each panel, allocate nBuffers buffers of nWords for calibrated data on the GPU
  // This space is organized so that for each buffer, the panel data is contiguous, i.e.,
  // calibBuffers[nBuffers][nPanels][nWords]
  chkError(cudaMalloc(&m_calibBuffers, nBuffers * nPanels * nWords * sizeof(*m_calibBuffers)));
}

void MemPoolGpu::destroyCalibBuffers()
{
  if (m_calibBuffers) {
    chkError(cudaFree(m_calibBuffers));
    m_calibBuffers = 0;
  }
}

void MemPoolGpu::createReduceBuffers(unsigned nBuffers, unsigned nWords)
{
  // Allocate buffers for reduced data on the GPU
  chkError(cudaMalloc(&m_dataBuffers, nBuffers * nWords * sizeof(*m_dataBuffers)));
}

void MemPoolGpu::destroyReduceBuffers()
{
  if (m_dataBuffers) {
    chkError(cudaFree(m_dataBuffers));
    m_dataBuffers = 0;
  }
}
