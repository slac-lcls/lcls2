#include "MemPool.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/range.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <cstdio>                       // For fopen, fgets, sscanf
#include <cstring>                      // For strncmp
#include <unistd.h>                     // For getuid, geteuid
#include <linux/capability.h>           // For CAP_SYS_ADMIN (no libcap needed)

using logging = psalg::SysLog;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


namespace Drp {
  namespace Gpu {

    [[maybe_unused]]
    static constexpr unsigned GPU_OFFSET       = GPU_ASYNC_CORE_OFFSET;
    static constexpr size_t   DMA_BUFFER_SIZE  = 64*1024; // Minimum buffer
    static constexpr unsigned DMA_BUFFER_COUNT = 4;       // Default
  } // Gpu
} // Drp

static void chkMemory(const void* pointer, unsigned count, size_t size, const char* name)
{
  if (!pointer) {
    logging::critical("cudaMalloc returned no memory for %u %s of size %zu\n", count, name, size);
    exit(-ENOMEM);
  }
}


DataDev::DataDev(const char* path)
{
  fd_ = open(path, O_RDWR);
  if (fd_ < 0) {
    logging::critical("Error opening %s: %m", path);
    abort();
  }
}


#ifndef HOST_REARMS_DMA                 // Only this build needs the privilege
// Report how the process came by the privilege that the mapping below needs, so
// that a log shows whether the intended mechanism is the one actually in force.
// Several mechanisms work, and they are not equally desirable: a leftover setuid
// bit or file capability on the executable would let the mapping succeed while
// granting far more than is wanted, and would do so silently.  See the comment in
// MemPool.hh for how the privilege is meant to be granted.
static void _reportPrivilege()
{
  // Read the effective capability set rather than link against libcap for one value
  uint64_t capEff{0};
  bool     capEffKnown{false};
  if (FILE* status = fopen("/proc/self/status", "r")) {
    char line[256];
    while (fgets(line, sizeof(line), status)) {
      if (strncmp(line, "CapEff:", 7) == 0) {
        capEffKnown = sscanf(line + 7, "%lx", &capEff) == 1;
        break;
      }
    }
    fclose(status);
  }
  bool const hasSysAdmin{capEffKnown && (capEff & (1UL << CAP_SYS_ADMIN))};

  logging::info("Privilege: uid %u, euid %u, CapEff 0x%016lx, CAP_SYS_ADMIN %s",
                getuid(), geteuid(), capEff,
                capEffKnown ? (hasSysAdmin ? "yes" : "no") : "unknown");

  if (geteuid() == 0) {
    logging::warning("Running with euid 0, which grants far more than the "
                     "CAP_SYS_ADMIN this needs.  If that was not intended, look for "
                     "a leftover setuid bit on the executable "
                     "('chmod u-s' to clear it); see the comment in MemPool.hh");
  } else if (hasSysAdmin) {
    logging::debug("Holding CAP_SYS_ADMIN without being root, as intended");
  }
}
#endif


MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool           (para),
  m_setMaskBytesDone(false),
  m_hostWrtBufsSize (0),
  m_calibBufsSize   (0),
  m_calibBuffers_d  (nullptr),
  m_reduceBufsSize  (0),
  m_reduceBuffers_d (nullptr)
{
  dmaBuffers = nullptr;                 // Unused: cause a crash if accessed

  // Determine DMA buffer size and round up to units of 64 kB for alignment
  // The DMA buffer size must include space for the TimingHeader
  if (para.kwargs.find("dmaBufSize") != para.kwargs.end())
    m_dmaSize = std::stoul(para.kwargs.at("dmaBufSize"));
  else
    m_dmaSize = DMA_BUFFER_SIZE;
  m_dmaSize = ((m_dmaSize >> 16) + (m_dmaSize & 0xffff ? 1 : 0)) << 16;

  // Determine DMA buffer count
  if (para.kwargs.find("dmaBufCount") != para.kwargs.end())
    m_dmaCount = std::stoul(para.kwargs.at("dmaBufCount"));
  else
    m_dmaCount = DMA_BUFFER_COUNT;
  if (m_dmaCount & (m_dmaCount-1)) { // GPU divides by non-powers-of-2 are expensive
    logging::critical("The number of DMA buffers must be a power of 2; got %u",
                      m_dmaCount);
    abort();
  }

  // Set up GPU
  unsigned gpuId = 0;
  if (para.kwargs.find("gpuId") != para.kwargs.end())
    gpuId = std::stoul(para.kwargs.at("gpuId"));

  // Initialize the device context
  if (!m_context.init(gpuId, para.verbose == 0)) {
    logging::critical("CUDA initialize failed");
    abort();
  }

  // Now that we _have_ the device, perhaps we now also need to _set_ it?
  chkError(cudaSetDevice(m_context.deviceNo()));

  // Check for required device attributes
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

  logging::debug("Done with device context setup\n");

  void* fpgaRegs{nullptr};

  // Check for normal DAQ mode (datadev device) or simulator mode (null device)
  if (para.device != "/dev/null") {
    m_panel = make_shared<DetPanel>(para.device);
    logging::info("PGP device '%s' opened", m_panel->name.c_str());
    const auto fd = m_panel->datadev.fd();

    // Check for GPUDirect support
    if (!gpuIsGpuAsyncSupported(fd)) {
      logging::critical("Firmware or driver does not support GPUAsync!");
      abort();
    }

    // Validate buffer count
    if (m_dmaCount > gpuGetMaxBuffers(fd)) {
      logging::critical("Too many buffers requested: %d > %d\n",
                        m_dmaCount, gpuGetMaxBuffers(fd));
      abort();
    }

    // Map device control registers to the host
    fpgaRegs = dmaMapRegister(fd, GPU_ASYNC_CORE_OFFSET, GPU_ASYNC_CORE_SIZE);
    if (!fpgaRegs) {
      logging::critical("Failed to map FPGA registers");
      abort();
    }

    // Init a register object
    m_panel->coreRegs.initialize(false, fpgaRegs);

#ifndef HOST_REARMS_DMA
    _reportPrivilege();

    // Map the GpuAsyncCore FPGA registers into the CUDA address space to allow the GPU to access them
    // CU_MEMHOSTREGISTER_IOMEMORY requires a privileged process; without the
    // privilege this fails with CUDA_ERROR_NOT_PERMITTED.  Report the CUresult
    // rather than errno, which has nothing to do with it, so that NOT_PERMITTED --
    // a privilege problem, which a capability may be enough to solve -- can be
    // told from NOT_SUPPORTED, which means the kernel or platform cannot do this
    // at all and no amount of privilege will help.
    if (chkError(cuMemHostRegister(fpgaRegs, GPU_ASYNC_CORE_SIZE,
                                   CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP),
                 "Mapping the GpuAsyncCore FPGA registers for GPU access failed")) {
      logging::info("The GPU rearms DMA buffers by writing these registers, which "
                    "needs CAP_SYS_ADMIN.  Grant it without running as root by "
                    "launching under 'setpriv --ambient-caps=-all,+sys_admin' -- see "
                    "the comment in MemPool.hh for the full command.  Failing that, "
                    "rebuild with HOST_REARMS_DMA defined in MemPool.hh to have the "
                    "CPU do the rearming, at the cost of a longer delay before a DMA "
                    "buffer can be reused.");
      abort();
    }

    logging::debug("Mapped FPGA registers");
#endif

    // Configure max. FPGA->GPU buffer on the FPGA side
    m_panel->coreRegs.setRemoteWriteMaxSize(0, m_dmaSize);

    // Get the DMA_AXI_CONFIG_G.DATA_BYTES_C from FPGA
    const uint32_t dmaHeaderSize = m_panel->coreRegs.dmaDataBytes();

    // Allocate DMA write buffers (FPGA->GPU)
    auto& dmaBufs_d = m_panel->dmaBuffers_d;
    chkError(cudaMalloc(&dmaBufs_d, m_dmaCount * sizeof(*dmaBufs_d)));
    m_panel->dmaBuffers.resize(m_dmaCount);
    for (unsigned i = 0; i < m_dmaCount; ++i) {
      uint8_t* dp{nullptr};
      size_t   sz{dmaHeaderSize + m_dmaSize};
      chkError(cudaMalloc(&dp,    sz));
      chkMemory          ( dp,    sz, sizeof(*dp), "dmaBuffers");
      chkError(cudaMemset( dp, 0, sz));
      m_panel->dmaBuffers[i] = dp;
      chkError(cudaMemcpy(&dmaBufs_d[i], &dp, sizeof(*dmaBufs_d), cudaMemcpyDefault));
    }

    // Map the GPU's DMA write buffers to the FPGA registers
    for (unsigned i = 0; i < m_dmaCount; ++i) {
      if (gpuAddNvidiaMemory(fd, 1, (uint64_t)m_panel->dmaBuffers[i], m_dmaSize)) {
        logging::critical("gpuAddNvidiaMemory failed: %m");
        abort();
      }
      logging::info("DMA buffer[%u] dptr %p, size %u, free list[%u] %p",
                    i, m_panel->dmaBuffers[i], m_dmaSize,
                    i, (uint8_t*)fpgaRegs + m_panel->coreRegs.freeListOffset(0) + i*4);
    }
  } else {                              // Simulator mode
    m_panel = std::make_shared<DetPanel>(para.device);
    logging::info("NULL PGP device '%s' opened", para.device.c_str());

    uint8_t* ptr{nullptr};
    size_t regBlkSize{0x600 * sizeof(uint32_t)};
    chkError(cudaHostAlloc(&ptr,    regBlkSize, cudaHostAllocDefault));
    chkMemory             ( ptr,    regBlkSize, sizeof(*ptr), "fpgaRegs");
    chkError(cudaMemset   ( ptr, 0, regBlkSize));

    fpgaRegs = ptr;

    // Init a register object
    m_panel->coreRegs.initialize(true, fpgaRegs);

    // Configure max. FPGA->GPU buffer on the "FPGA" side
    m_panel->coreRegs.setRemoteWriteMaxSize(0, m_dmaSize);

    m_panel->coreRegs.setDataBytes(32);  // Fake up DMA frame size

    // Get the DMA_AXI_CONFIG_G.DATA_BYTES_C from "FPGA"
    const uint32_t dmaHeaderSize = m_panel->coreRegs.dmaDataBytes();

    // Allocate "DMA" write buffers on the GPU
    auto& dmaBufs_d = m_panel->dmaBuffers_d;
    chkError(cudaMalloc(&dmaBufs_d, m_dmaCount * sizeof(*dmaBufs_d)));
    m_panel->dmaBuffers.resize(m_dmaCount);
    for (unsigned i = 0; i < m_dmaCount; ++i) {
      uint8_t* dp{nullptr};
      size_t   sz{dmaHeaderSize + m_dmaSize};
      chkError(cudaMalloc(&dp,    sz));
      chkMemory          ( dp,    sz, sizeof(*dp), "dmaBuffers");
      chkError(cudaMemset( dp, 0, sz));
      m_panel->dmaBuffers[i] = dp;
      chkError(cudaMemcpy(&dmaBufs_d[i], &dp, sizeof(*dmaBufs_d), cudaMemcpyDefault));
      logging::info("DMA buffer[%u] dptr %p, size %u, free list[%u] %p",
                    i, m_panel->dmaBuffers[i], m_dmaSize,
                    i, (uint8_t*)fpgaRegs + m_panel->coreRegs.freeListOffset(0) + i*4);
    }

    // No need to call setMaskBytes, so fake done
    m_setMaskBytesDone = true;
  }

  logging::debug("Done with device mem alloc for %s, FPGA regs %p\n",
                 para.device.c_str(), fpgaRegs);

  // Stop the FPGA side
  m_panel->coreRegs.setWriteEnable(0);

  // Ensure that timing messages are DMAed to the GPU
  dmaTgtSet(m_panel->coreRegs, DmaTgt_t::TGT_GPU);

  // Initialize the base class before using dependencies like nbuffers()
  _initialize(para);
  pgpEvents.resize(nbuffers()); // Need 1 per intermediate buffer - w/o this have only dmaCount buffers

  // Set up intermediate buffer pointers
  chkError(cudaMalloc(&m_reduceBuffers_d,    nbuffers() * sizeof(*m_reduceBuffers_d)));
  chkMemory          ( m_reduceBuffers_d,    nbuffers(),  sizeof(*m_reduceBuffers_d), "reduceBuffers");
  chkError(cudaMemset( m_reduceBuffers_d, 0, nbuffers() * sizeof(*m_reduceBuffers_d)));

  logging::debug("Done with setting up memory\n");
}

MemPoolGpu::~MemPoolGpu()
{
  // Release the memory held by the driver
  if ((gpuRemNvidiaMemory(m_panel->datadev.fd())) < 0)
    logging::error("gpuRemNvidiaMemory failed: %m");

  // Free the DMA buffers
  for (unsigned i = 0; i < dmaCount(); ++i) {
    if (m_panel->dmaBuffers[i])  chkError(cudaFree(m_panel->dmaBuffers[i]));
    m_panel->dmaBuffers[i] = nullptr;
  }
  if (m_panel->dmaBuffers_d)  chkError(cudaFree(m_panel->dmaBuffers_d));
  m_panel->dmaBuffers_d = nullptr;

  // Free the intermediate buffers
  destroyReduceBuffers();
  destroyCalibBuffers();
  destroyHostBuffers();

  m_panel.reset();
}

int MemPoolGpu::setMaskBytes(uint8_t laneMask, unsigned virtChan)
{
  int retval = 0;
  if (m_setMaskBytesDone) {
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
    if (dmaSetMaskBytes(m_panel->datadev.fd(), mask)) {
      retval = 1; // error
    } else {
      m_setMaskBytesDone = true;
    }
  }
  return retval;
}


void MemPoolGpu::createHostBuffers(size_t size)
{
  if (m_hostWrtBufsSize) {
    logging::error("Attempt to reallocate HostWrtBuffers");
    return;
  }

  // Allocate buffers for the DMA descriptors, TimingHeaders and TEB input data
  // in pinned memory and make them visible on both the CPU and the GPU
  auto nBufs = nbuffers();
  chkError(cudaHostAlloc(&m_hostWrtBufs,    nBufs * size, cudaHostAllocDefault));
  chkMemory             ( m_hostWrtBufs,    nBufs,  size, "hostWrtBufsVec_h");
  chkError(cudaMemset   ( m_hostWrtBufs, 0, nBufs * size)); // Avoid reading stale data on re-Configure

  m_hostWrtBufsSize = size;

  auto sz = size / sizeof(*m_hostWrtBufs);
  logging::info("Host write buffers: %p : %p, size %u * %zu B\n",
                &m_hostWrtBufs[0], &m_hostWrtBufs[(nBufs-1) * sz], nBufs, size);
}

void MemPoolGpu::destroyHostBuffers()
{
  if (m_hostWrtBufsSize) {
    chkError(cudaFreeHost(m_hostWrtBufs));
    m_hostWrtBufs = nullptr;
    m_hostWrtBufsSize = 0;
  }
}

void MemPoolGpu::createCalibBuffers(unsigned nElements)
{
  if (m_calibBufsSize) {
    logging::error("Attempt to reallocate CalibBuffers");
    return;
  }

  // Allocate nBufs buffers of nElements for calibrated data on the GPU
  // This space is organized as calibBuffers[nBufs][nElements]
  auto nBufs = nbuffers();
  auto size  = nElements * sizeof(*m_calibBuffers_d);
  chkError(cudaMalloc(&m_calibBuffers_d,    nBufs * size));
  chkMemory          ( m_calibBuffers_d,    nBufs,  size, "calibBuffers_d");
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

  // Allocate nBufs buffers for reduced data on the GPU,
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
