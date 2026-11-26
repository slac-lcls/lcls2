#include "MemPool.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/range.hh"
#include "psdaq/service/Affinity.hh"   // logCpuAllocation, logDeviceLocality
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <cstdio>                       // For fopen, fgets, sscanf
#include <cstring>                      // For strncmp
#include <new>                          // For placement new, over pinned memory
#include <cerrno>                       // For errno
#include <unistd.h>                     // For getuid, geteuid, syscall
#include <linux/capability.h>           // For CAP_SYS_ADMIN (no libcap needed)
#include <sys/prctl.h>                  // For PR_CAP_AMBIENT
#include <sys/syscall.h>                // For SYS_capget, SYS_capset

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
    static constexpr size_t   GPU_PAGE_SIZE    = 1ul<<16; // GPU_BOUND_SHIFT in the driver
  } // Gpu
} // Drp

static void chkMemory(const void* pointer, unsigned count, size_t size, const char* name)
{
  if (!pointer) {
    logging::critical("cudaMalloc returned no memory for %u %s of size %zu\n", count, name, size);
    exit(-ENOMEM);
  }
}


// Allocate GPU memory starting on a GPU page boundary, which gpuAddNvidiaMemory()
// requires ("memory must be aligned to GPU page boundary"); cudaMalloc guarantees only
// 256 or 512 B.  'raw' is for cudaFree; the return value is for the FPGA and kernels.
static uint8_t* _allocAlignedDma(uint8_t*& raw, size_t size, const char* name)
{
  const size_t over{size + Gpu::GPU_PAGE_SIZE - 1};
  raw = nullptr;
  chkError(cudaMalloc(&raw, over));
  chkMemory          ( raw, over, sizeof(*raw), name);
  chkError(cudaMemset( raw, 0, over));
  auto addr = reinterpret_cast<uintptr_t>(raw);
  return reinterpret_cast<uint8_t*>((addr + Gpu::GPU_PAGE_SIZE - 1) & ~(Gpu::GPU_PAGE_SIZE - 1));
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
// Read the effective capability set rather than link against libcap for one value.
// Returns false when it cannot be determined, leaving capEff untouched.
static bool _capEff(uint64_t& capEff)
{
  bool known{false};
  if (FILE* status = fopen("/proc/self/status", "r")) {
    char line[256];
    while (fgets(line, sizeof(line), status)) {
      if (strncmp(line, "CapEff:", 7) == 0) {
        known = sscanf(line + 7, "%lx", &capEff) == 1;
        break;
      }
    }
    fclose(status);
  }
  return known;
}

// Log how the process got the mapping's privilege, so a leftover setuid bit, which
// grants far more than needed, doesn't go unnoticed.
static void _reportPrivilege()
{
  uint64_t capEff{0};
  bool const capEffKnown{_capEff(capEff)};
  bool const hasSysAdmin{capEffKnown && (capEff & (1UL << CAP_SYS_ADMIN))};

  logging::info("Privilege: uid %u, euid %u, CapEff 0x%016lx, CAP_SYS_ADMIN %s",
                getuid(), geteuid(), capEff,
                capEffKnown ? (hasSysAdmin ? "yes" : "no") : "unknown");

  if (geteuid() == 0) {
    // Likely a leftover setuid bit on the executable ('chmod u-s' clears it)
    logging::warning("Running as euid 0, far more privilege than CAP_SYS_ADMIN");
  } else if (hasSysAdmin) {
    logging::debug("Holding CAP_SYS_ADMIN without being root, as intended");
  }
}


// Give up CAP_SYS_ADMIN once the registers are mapped.  That mapping, made once here,
// is drp_gpu's only privileged call; datadev's ioctls check thread-group ownership,
// not capabilities.  Raw syscalls: one capability isn't worth a libcap dependency.
static void _dropPrivilege()
{
  // Ambient first, explicitly, though capset() below would clear it too.  EINVAL
  // means a kernel without ambient capabilities.
  if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) && errno != EINVAL) {
    logging::warning("Could not clear the ambient capability set: %m");
  }

  __user_cap_header_struct hdr{_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct   data[2]{};
  if (syscall(SYS_capget, &hdr, data)) {
    logging::warning("capget failed (%m); CAP_SYS_ADMIN not dropped");
    return;
  }

  // Drop it from permitted as well as effective, so it cannot be raised again
  auto const idx  = CAP_TO_INDEX(CAP_SYS_ADMIN);
  auto const mask = CAP_TO_MASK(CAP_SYS_ADMIN);
  data[idx].effective   &= ~mask;
  data[idx].permitted   &= ~mask;
  data[idx].inheritable &= ~mask;

  if (syscall(SYS_capset, &hdr, data)) {
    logging::warning("capset failed (%m); CAP_SYS_ADMIN is still held");
    return;
  }

  // Confirm rather than assume: a silent no-op here would leave the privilege held
  // for the whole run, which is the thing this is meant to avoid.
  uint64_t capEff{0};
  if (_capEff(capEff)) {
    if (capEff & (1UL << CAP_SYS_ADMIN)) {
      logging::warning("CAP_SYS_ADMIN is still effective after capset; CapEff "
                       "0x%016lx", capEff);
    } else {
      logging::info("Dropped CAP_SYS_ADMIN; CapEff now 0x%016lx", capEff);
    }
  }

  if (geteuid() == 0) {
    // The kernel restores a root process's capabilities across an exec
    logging::warning("Still euid 0: dropping CAP_SYS_ADMIN does not drop root's privileges");
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
  m_reduceBufsRsvd  (0),
  m_reduceBufsRaw   (0),
  m_reduceBuffers_d (nullptr),
  m_trBufsSize      (0),
  m_trBufCnt        (0),
  m_trBuffers_d     (nullptr),
  m_prescaleBufsRsvd     (0),
  m_prescaleBufsRaw      (0),
  m_prescaleBufsDscr     (0),
  m_prescaleBufsSize     (0),
  m_prescaleBufCnt       (0),
  m_prescaleBuffers_d    (nullptr),
  m_rawTicket_d     (nullptr),
  m_rawOverflow     (nullptr),
  m_rawReleased     (nullptr)
{
  dmaBuffers = nullptr;                 // Unused: cause a crash if accessed

  // Determine DMA buffer size and round up to a whole number of GPU pages
  // The DMA buffer size must include space for the TimingHeader
  if (para.kwargs.find("dmaBufSize") != para.kwargs.end())
    m_dmaSize = std::stoul(para.kwargs.at("dmaBufSize"));
  else
    m_dmaSize = DMA_BUFFER_SIZE;
  // The driver wants the size, as well as the address, a multiple of GPU_PAGE_SIZE
  m_dmaSize = ((m_dmaSize + Gpu::GPU_PAGE_SIZE - 1) / Gpu::GPU_PAGE_SIZE) * Gpu::GPU_PAGE_SIZE;

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

  // Report the CPUs granted and whether the datadev and GPU are local to them
  logCpuAllocation("MemPoolGpu");
  logDeviceLocality("MemPoolGpu", "GPU",     pciAddrOfGpu(m_context.deviceNo()));
  logDeviceLocality("MemPoolGpu", "datadev", pciAddrOfDatadev(para.device));

  // DMA goes card -> GPU, so their being on different NUMA nodes costs every event
  int gpuNuma = numaNodeOfGpu(m_context.deviceNo());
  setHostAllocNode(gpuNuma);
  int dmaNuma = numaNodeOfDatadev(para.device);
  if (gpuNuma >= 0 && dmaNuma >= 0 && gpuNuma != dmaNuma)
    logging::warning("MemPoolGpu: datadev is on NUMA node %d but its GPU is on %d, "
                     "so DMA crosses the socket link", dmaNuma, gpuNuma);

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

    // Map the GpuAsyncCore FPGA registers into the CUDA address space to allow the GPU to access them.
    // Report the CUresult, not errno: NOT_PERMITTED is a privilege problem, while
    // NOT_SUPPORTED means no privilege will help.
    if (chkError(cuMemHostRegister(fpgaRegs, GPU_ASYNC_CORE_SIZE,
                                   CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP),
                 "Mapping the GpuAsyncCore FPGA registers for GPU access failed")) {
      // The GPU rearms DMA buffers through these registers.  Without the capability,
      // only a build with HOST_REARMS_DMA (MemPool.hh) can run, more slowly.
      logging::critical("Mapping the GpuAsyncCore registers needs CAP_SYS_ADMIN");
      abort();
    }

    logging::debug("Mapped FPGA registers");

    // That was the only call needing it, so hold it no longer
    _dropPrivilege();
#endif

    // Configure max. FPGA->GPU buffer on the FPGA side
    m_panel->coreRegs.setRemoteWriteMaxSize(0, m_dmaSize);

    // Get the DMA_AXI_CONFIG_G.DATA_BYTES_C from FPGA
    const uint32_t dmaHeaderSize = m_panel->coreRegs.dmaDataBytes();

    // Allocate DMA write buffers (FPGA->GPU)
    auto& dmaBufs_d = m_panel->dmaBuffers_d;
    chkError(cudaMalloc(&dmaBufs_d, m_dmaCount * sizeof(*dmaBufs_d)));
    m_panel->dmaBuffers.resize(m_dmaCount);
    m_panel->dmaRawPtrs.resize(m_dmaCount);
    for (unsigned i = 0; i < m_dmaCount; ++i) {
      size_t   sz{dmaHeaderSize + m_dmaSize};
      uint8_t* dp{_allocAlignedDma(m_panel->dmaRawPtrs[i], sz, "dmaBuffers")};
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
    chkError(hostAlloc(&ptr, regBlkSize));
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
    m_panel->dmaRawPtrs.resize(m_dmaCount);
    for (unsigned i = 0; i < m_dmaCount; ++i) {
      size_t   sz{dmaHeaderSize + m_dmaSize};
      uint8_t* dp{_allocAlignedDma(m_panel->dmaRawPtrs[i], sz, "dmaBuffers")};
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
  // One per staging buffer, not per f/w DMA buffer.  Must follow _initialize(), which
  // sets m_nbuffers and sizes pgpEvents to the smaller dmaCount.
  pgpEvents.resize(nbuffers());

  chkError(cudaMalloc(&m_reduceBuffers_d,    nbuffers() * sizeof(*m_reduceBuffers_d)));
  chkMemory          ( m_reduceBuffers_d,    nbuffers(),  sizeof(*m_reduceBuffers_d), "reduceBuffers");
  chkError(cudaMemset( m_reduceBuffers_d, 0, nbuffers() * sizeof(*m_reduceBuffers_d)));

  logging::debug("Done with setting up memory\n");
}

MemPoolGpu::~MemPoolGpu()
{
  // Release the memory held by the driver; the simulator's /dev/null holds none
  if (m_panel->name != "/dev/null" && gpuRemNvidiaMemory(m_panel->datadev.fd()) < 0)
    logging::error("gpuRemNvidiaMemory failed: %m");

  // Free the DMA buffers
  for (unsigned i = 0; i < dmaCount(); ++i) {
    // dmaRawPtrs, not dmaBuffers: the latter is an aligned pointer *into* the allocation.
    if (m_panel->dmaRawPtrs[i])  chkError(cudaFree(m_panel->dmaRawPtrs[i]));
    m_panel->dmaRawPtrs[i]  = nullptr;
    m_panel->dmaBuffers[i] = nullptr;
  }
  if (m_panel->dmaBuffers_d)  chkError(cudaFree(m_panel->dmaBuffers_d));
  m_panel->dmaBuffers_d = nullptr;

  destroyTransitionBuffers();
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
  chkError(hostAlloc(&m_hostWrtBufs, nBufs * size));
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
    chkError(hostFree(m_hostWrtBufs));
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

void MemPoolGpu::createReduceBuffers(size_t nBytes, size_t reserved, size_t rawBytes)
{
  if (m_reduceBufsSize) {
    logging::error("Attempt to reallocate ReduceBuffers");
    return;
  }

  // Round each region to whole uint64_ts, raw separately so the header abutting it
  // stays aligned
  nBytes   = sizeof(uint64_t)*((nBytes   + sizeof(uint64_t)-1)/sizeof(uint64_t));
  reserved = sizeof(uint64_t)*((reserved + sizeof(uint64_t)-1)/sizeof(uint64_t));
  rawBytes = sizeof(uint64_t)*((rawBytes + sizeof(uint64_t)-1)/sizeof(uint64_t));

  // Each buffer is
  //
  //   [ hdr reserve ][ raw reserve ][ reduced payload ]
  //                                 ^ m_reduceBuffers_d
  //
  // The recorder grows *backwards* from m_reduceBuffers_d, so unused reserve never
  // reaches the file, and Reducers index the payload without knowing about raw.
  uint8_t* reduceBufferBase;
  auto   nBufs = nbuffers();
  size_t size  = (nBytes + reserved + rawBytes) * sizeof(*m_reduceBuffers_d);
  chkError(cudaMalloc(&reduceBufferBase,    nBufs * size));
  chkMemory          ( reduceBufferBase,    nBufs,  size, "reduceBufferBase");
  chkError(cudaMemset( reduceBufferBase, 0, nBufs * size));
  m_reduceBuffers_d = reduceBufferBase + reserved + rawBytes;

  m_reduceBufsSize = nBytes;           // Doesn't include the reserved portions!
  m_reduceBufsRsvd = reserved;
  m_reduceBufsRaw  = rawBytes;

  auto sz = size / sizeof(*m_reduceBuffers_d);
  logging::info("Reduce buffers: [base %p] %p : %p, size %u * (%zu + %zu + %zu) B\n", reduceBufferBase,
                &m_reduceBuffers_d[0], &m_reduceBuffers_d[(nBufs-1) * sz], nBufs, reserved, rawBytes, nBytes);
}

void MemPoolGpu::destroyReduceBuffers()
{
  if (m_reduceBufsSize) {
    chkError(cudaFree(m_reduceBuffers_d - m_reduceBufsRsvd - m_reduceBufsRaw));
    m_reduceBufsSize = 0;
    m_reduceBufsRsvd = 0;
    m_reduceBufsRaw  = 0;
  }
}

void MemPoolGpu::createTransitionBuffers(size_t nBytes, unsigned nBufs)
{
  if (m_trBufsSize) {
    logging::error("Attempt to reallocate TransitionBuffers");
    return;
  }

  // Round up for buffer alignment, as the other regions do
  nBytes = sizeof(uint64_t)*((nBytes + sizeof(uint64_t)-1)/sizeof(uint64_t));

  // Separate from the reduce buffers so the largest transition isn't multiplied by
  // nbuffers().  A slot per transition, as Drp::MemPool's m_transitionBuffers, and no
  // reserve: the whole datagram is copied from the host.
  auto size = nBufs * nBytes;
  chkError(cudaMalloc(&m_trBuffers_d,    size));
  chkMemory          ( m_trBuffers_d,    nBufs, nBytes, "transitionBuffers");
  chkError(cudaMemset( m_trBuffers_d, 0, size));

  m_trBufsSize = nBytes;
  m_trBufCnt   = nBufs;

  logging::info("Transition buffers: %p : %p, size %u * %zu B\n",
                &m_trBuffers_d[0], &m_trBuffers_d[(nBufs-1) * nBytes], nBufs, nBytes);
}

void MemPoolGpu::destroyTransitionBuffers()
{
  if (m_trBufsSize) {
    chkError(cudaFree(m_trBuffers_d));
    m_trBuffers_d = nullptr;
    m_trBufsSize  = 0;
    m_trBufCnt    = 0;
  }
}

void MemPoolGpu::createPrescaleBuffers(size_t nBytes, unsigned nBufs,
                                  size_t hdrBytes, size_t dscrBytes)
{
  if (m_prescaleBufsRaw) {
    logging::error("Attempt to reallocate prescale buffers");
    return;
  }

  // A prescaled event's whole datagram, so the raw block isn't multiplied by nbuffers():
  //
  //   [ Dgram ][ raw descr ][ raw ][ reduced descr ][ reduced copy ]
  //
  // hdrBytes is the Dgram plus the raw descriptors; dscrBytes, the reduced ones, sits
  // BETWEEN the payloads (CreateData writes descriptors just before their bytes).
  auto reserved = hdrBytes;
  auto redSize  = reduceBufsSize();
  nBytes    = sizeof(uint64_t)*((nBytes    + sizeof(uint64_t)-1)/sizeof(uint64_t));
  reserved  = sizeof(uint64_t)*((reserved  + sizeof(uint64_t)-1)/sizeof(uint64_t));
  dscrBytes = sizeof(uint64_t)*((dscrBytes + sizeof(uint64_t)-1)/sizeof(uint64_t));
  redSize   = sizeof(uint64_t)*((redSize   + sizeof(uint64_t)-1)/sizeof(uint64_t));

  auto stride = reserved + nBytes + dscrBytes + redSize;
  chkError(cudaMalloc(&m_prescaleBuffers_d,    nBufs * stride));
  chkMemory          ( m_prescaleBuffers_d,    nBufs,  stride, "rawBuffers");
  chkError(cudaMemset( m_prescaleBuffers_d, 0, nBufs * stride));

  m_prescaleBufsRsvd = reserved;
  m_prescaleBufsRaw  = nBytes;
  m_prescaleBufsDscr = dscrBytes;
  m_prescaleBufsSize = redSize;
  m_prescaleBufCnt   = nBufs;

  // The ticket is advanced by the _event kernel; released by the recorder.  Released
  // and overflow are pinned so that each side sees the other's stores without a copy;
  // being atomic is what makes that visibility guaranteed rather than hoped for.
  chkError(cudaMalloc    (&m_rawTicket_d,    sizeof(*m_rawTicket_d)));
  chkError(cudaMemset    ( m_rawTicket_d, 0, sizeof(*m_rawTicket_d)));
  chkError(hostAlloc(&m_rawOverflow, sizeof(*m_rawOverflow)));
  chkError(hostAlloc(&m_rawReleased, sizeof(*m_rawReleased)));
  new (m_rawOverflow) cuda::std::atomic<unsigned>{0};
  new (m_rawReleased) cuda::std::atomic<unsigned>{0};

  logging::info("Prescale buffers: %p : %p, size %u * (%zu + %zu + %zu + %zu) B\n",
                &m_prescaleBuffers_d[0], &m_prescaleBuffers_d[(nBufs-1) * stride],
                nBufs, reserved, nBytes, dscrBytes, redSize);
}

void MemPoolGpu::destroyPrescaleBuffers()
{
  if (m_prescaleBufsRaw) {
    chkError(cudaFree(m_prescaleBuffers_d));
    if (m_rawTicket_d)  chkError(cudaFree(m_rawTicket_d));
    if (m_rawOverflow)  chkError(hostFree(m_rawOverflow));
    if (m_rawReleased)  chkError(hostFree(m_rawReleased));
    m_prescaleBuffers_d = nullptr;
    m_rawTicket_d  = nullptr;
    m_rawOverflow  = nullptr;
    m_rawReleased  = nullptr;
    m_prescaleBufsRsvd  = 0;
    m_prescaleBufsRaw   = 0;
    m_prescaleBufsDscr  = 0;
    m_prescaleBufsSize  = 0;
    m_prescaleBufCnt    = 0;
  }
}
