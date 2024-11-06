#include "GpuWorker_impl.hh"

#include <GpuAsync.h>
#include "drp.hh"
// @todo: Revisit: #include "GpuDetector.hh"
#include "Detector.hh"
#include "spscqueue.hh"

#include <cuda_runtime.h>

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"

#include <thread>

using namespace Drp;
using namespace Pds;
using namespace XtcData;
using logging = psalg::SysLog;


#define EMPTY ""           // Ensures there is an arg when __VA_ARGS__ is blank
#define chkFatal(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, true,  EMPTY __VA_ARGS__)
#define chkError(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, false, EMPTY __VA_ARGS__)

namespace Drp {

static
bool checkError(CUresult status, const char* func, const char* file, int line, bool crash=true, const char* msg="")
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

static
bool checkError(cudaError status, const char* func, const char* file, int line, bool crash=true, const char* msg="")
{
  if (status != cudaSuccess) {
    logging::error("%s:%d:  '%s'\n  %s\n  status %d: info: %s - %s\n", file, line, func, status, cudaGetErrorString(status), msg);
    if (crash)  abort();
    return true;
  }
  return false;
}

struct DmaDsc
{
  int32_t  ret;
  uint32_t size;                        // @todo: Guess
  uint32_t index;
  uint32_t dest;
  uint32_t flags;
  uint32_t errors;
  uint32_t _rsvd[2];                    // @todo: ???
};

};

CudaContext::CudaContext()
{
  chkFatal(cuInit(0), "Error while initting cuda");
}

bool CudaContext::initialize(int device)
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
    logging::error(
                   "  Consider setting NVreg_EnableStreamMemOPs=1 when loading the NVIDIA kernel module, "
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


void CudaContext::listDevices() {
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


GpuMemPool::GpuMemPool(const Parameters& para, MemPool& pool) :
  dmaBuffers(MAX_BUFFERS), // @todo: Revisit nbuffs and size
  m_pool    (pool)
{
}

GpuMemPool::~GpuMemPool()
{
  for (CUdeviceptr& dmaBuffer : dmaBuffers) {
    _gpuUnmapFpgaMem(dmaBuffer);
  }
  dmaBuffers.clear();

  ssize_t rc;
  if ((rc = gpuRemNvidiaMemory(fd())) < 0)
    logging::error("gpuRemNvidiaMemory failed: %zd: %M", rc);
}

int GpuMemPool::initialize()
{
  // Clear out any left-overs from last time
  int res = gpuRemNvidiaMemory(fd());
  if (res < 0)  logging::error("Error in gpuRemNvidiaMemory\n");
  logging::debug("Done with gpuRemNvidiaMemory() cleanup\n");

  // Allocate buffers on the GPU
  // This handles allocating buffers on the device and registering them with the driver.
  auto size = dmaSize();
  for (CUdeviceptr& dmaBuffer : dmaBuffers) {
    if (_gpuMapFpgaMem(dmaBuffer, 0, size, 1) != 0) {
      logging::error("Failed to alloc buffer list at number %zd",
                     &dmaBuffer - &dmaBuffers[0]);
      return -1;
    }
  }
  logging::debug("Done with device mem alloc\n");

  return 0;
}

// To avoid including drp.hh in GpuWorker_impl.hh:
unsigned               GpuMemPool::count()     const { return dmaBuffers.size(); }
size_t                 GpuMemPool::dmaSize()   const { return m_pool.dmaSize(); }
unsigned               GpuMemPool::nbuffers()  const { return m_pool.nbuffers(); }
int                    GpuMemPool::fd()        const { return m_pool.fd(); }
std::vector<PGPEvent>& GpuMemPool::pgpEvents() const { return m_pool.pgpEvents; }
Pebble&                GpuMemPool::pebble()    const { return m_pool.pebble; }
unsigned               GpuMemPool::allocate()        { return m_pool.allocate(); }

int GpuMemPool::_gpuMapFpgaMem(CUdeviceptr& buffer, uint64_t offset, size_t size, int write)
{
  auto idx = &buffer - &dmaBuffers[0];

  if (chkError(cuMemAlloc(&buffer, size))) {
    return -1;
  }
  cuMemsetD8(buffer, 0, size);

  int flag = 1;
  // This attribute is required for peer shared memory. It will synchronize every synchronous memory operation on this block of memory.
  if (chkError(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, buffer))) {
    cuMemFree(buffer);
    return -1;
  }

  if (gpuAddNvidiaMemory(fd(), write, buffer, size) < 0) {
    logging::error("gpuAddNvidiaMemory failed for buffer %zd", idx);
    cuMemFree(buffer);
    return -1;
  }

  return 0;
}

void GpuMemPool::_gpuUnmapFpgaMem(CUdeviceptr& buffer)
{
  chkError(cuMemFree(buffer));

  // FIXME: gpuOnly memory cannot be unmapped?
}


GpuWorker_impl::GpuWorker_impl(const Parameters& para, MemPool& pool, Detector& det) :
  m_det       (det),
  m_pool      (para, pool),
  m_streams   (m_pool.count()),
  m_batchStart(0),
  m_batchSize (0),
  m_dmaIndex  (0),
  m_para      (para)
{
  ////////////////////////////////////////////
  // Setup GPU
  ////////////////////////////////////////////

  if (para.verbose)
    m_context.listDevices();

  unsigned gpuId = 0;
  if (para.kwargs.find("gpuId") != para.kwargs.end())
    gpuId = std::stoul(const_cast<Parameters&>(para).kwargs["gpuId"]);

  if (!m_context.initialize(gpuId)) {
    logging::critical("CUDA initialize failed");
    abort();
  }
  logging::debug("Done with context setup\n");

  ////////////////////////////////////
  // Setup memory
  ////////////////////////////////////

  if (m_pool.initialize() < 0) {
    logging::critical("Error setting up memory");
    abort();
  }
  logging::debug("Done with setting up memory\n");

  ////////////////////////////////////
  // Allocate streams
  ////////////////////////////////////

  /** Allocate a stream per buffer **/
  for (auto& stream : m_streams) {
    chkFatal(cudaStreamCreate(&stream), "Error creating streams");
  }
  logging::debug("Done with creating streams\n");
}

GpuWorker::DmaMode_t GpuWorker_impl::dmaMode() const
{
  // @todo: This line addresses only lane 0
  uint32_t regVal;
  dmaReadRegister(m_pool.fd(), 0x00d0002c, &regVal);

  GpuWorker::DmaMode_t mode;
  switch (regVal)
  {
    case CPU:  mode = CPU;  break;
    case GPU:  mode = GPU;  break;
    default:   mode = ERR;  break;
  }
  return mode;
}

void GpuWorker_impl::dmaMode(GpuWorker::DmaMode_t mode_)
{
  // @todo: This line addresses only lane 0
  dmaWriteRegister(m_pool.fd(), 0x00d0002c, mode_);
}

void GpuWorker_impl::start(SPSCQueue<Batch>& collectorGpuQueues)
{
  // Launch one thread per stream
  for (unsigned i = 0; i < m_streams.size(); ++i) {
    m_threads.emplace_back(&GpuWorker_impl::_reader, std::ref(*this),
                           i, std::ref(collectorGpuQueues));
  }
}

void GpuWorker_impl::stop()
{
  // Stop and clean up the threads
  for (unsigned i = 0; i < m_threads.size(); ++i) {
    // @todo: Need to trigger the streams here
    m_threads[i].join();
  }
}

// @todo: Do we still want these?
//void GpuWorker_impl::timingHeaders(unsigned index, TimingHeader* buffer)
//{
//  auto idx = index & (m_streams.size() - 1);
//  chkFatal(cuMemcpyDtoH((void*)buffer, m_pool.dmaBuffers[idx], sizeof(*buffer)));
//}
//
//// @todo: This method is called when it has been recognized that data
////        has been DMAed into GPU memory and is ready to be processed
//void GpuWorker_impl::process(Batch& batch, bool& sawDisable)
//{
//  // Set up a buffer pool for timing headers visible to the host
//  // memcpy timing headers from device into this host pool
//  // memcpy the TEB input data right after the timing header?
//  //   Or put them in a separate pool?
//  // Form a batch of them
//  // Return from this routine when batch is full or Disable is seen
//
//}

// @todo: This method is called to wait for data to be DMAed into GPU memory
//        This method must then do several things:
//        - Find and copy the TimingHeader to a host buffer and share that
//          buffer's index
//        - Do the equivalent of the det.event() and det.slowUpdate() routines
//          to reorganize the data and prepare the Xtc header
//        - Prepare the TEB input data

void GpuWorker_impl::_reader(unsigned dmaIdx, SPSCQueue<Batch>& collectorGpuQueue)
{
  logging::info("GpuWorker::_reader[%d] starting\n", dmaIdx);

  size_t    size         = sizeof(DmaDsc)+sizeof(TimingHeader); //m_pool.dmaSize();
  uint32_t* hostWriteBuf = (uint32_t*)malloc(size);

  const uint32_t bufferMask = m_pool.nbuffers() - 1;

  // Handle L1Accepts, SlowUpdates and Disable
  bool        full       = false;
  bool        sawDisable = false;
  unsigned    last       = 0;
  unsigned    pgpIndex;
  const auto& stream     = m_streams[dmaIdx];
  auto        dmaBuffer  = m_pool.dmaBuffers[dmaIdx];
  while (true) {

    // Clear the GPU memory handshake space to zero
    logging::debug("%d clear memory\n", dmaIdx);
    chkFatal(cuStreamWriteValue32(stream, dmaBuffer + 4, 0x00, 0));

    // Write to the DMA start register in the FPGA
    logging::debug("Trigger write to buffer %d\n", dmaIdx);
    chkError(cuStreamSynchronize(stream));
    //auto rc = gpuSetWriteEn(m_pool.fd(), dmaIdx);
    auto rc = dmaWriteRegister(m_pool.fd(), 0xD00300 + 4 * dmaIdx, 1);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %d for write: %zd\n", dmaIdx, rc);
      perror("gpuSetWriteEn");
      abort();
    }

    // Spin on the handshake location until the value is greater than or equal to 1
    // This waits for the data to arrive in the GPU before starting the processing
    logging::debug("%d Wait memory value\n", dmaIdx);
    chkFatal(cuStreamWaitValue32(stream, dmaBuffer + 4, 0x1, CU_STREAM_WAIT_VALUE_GEQ));
    chkError(cuStreamSynchronize(stream));
    logging::debug("%d Done waiting\n", dmaIdx);
    unsigned nDmaRet = 1;  //*((unsigned*)(dmaBuffer + 4));

    chkError(cudaMemcpyAsync(hostWriteBuf, (void*)dmaBuffer, sizeof(DmaDsc)+sizeof(TimingHeader), cudaMemcpyDeviceToHost, stream));
    cuStreamSynchronize(stream);
    logging::debug("%d DtoH done\n", dmaIdx);

    auto dsc = (DmaDsc*)hostWriteBuf;
    logging::debug("*** %d hdr: ret %08x,  sz %08x, idx %08x, dst %08x, flg %08x, err %08x, rsvd %08x %08x\n",
                   dmaIdx, dsc->ret, dsc->size, dsc->index, dsc->dest, dsc->flags, dsc->errors,
                   dsc->_rsvd[0], dsc->_rsvd[1]);
    auto th  = (TimingHeader*)&hostWriteBuf[8];
    logging::debug("**G %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x\n",
                   dmaIdx, th->control(), th->pulseId(), th->time.value(), th->env, th->evtCounter,
                   th->_opaque[0], th->_opaque[1]);

    // @todo: Need indices, errors, etc., like from dmaBulkReadDmaIndex()
    // @todo: Handle multiple lanes
    uint32_t size = dsc->size;
    uint32_t lane = (dsc->dest >> 8) & 7;
    m_dmaSize   = size;
    m_dmaBytes += size;
    // @todo: Is this the case here also?
    // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
    // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
    // which thus won't have the expected header.  Take the exact match as an overflow indicator.
    if (size == m_pool.dmaSize()) {
      logging::critical("%d DMA overflowed buffer: %d vs %d", dmaIdx, size, m_pool.dmaSize());
      abort();
    }

    // @todo: dsc->index is always 0?
    //if (dmaIdx != dsc->index)
    //  logging::error("DMA index mismatch: got %u, expected %u\n",
    //                 dsc->index, dmaIdx);
    pgpIndex = th->evtCounter & bufferMask;
    if (pgpIndex != m_batchStart + last)
      logging::error("%d Event counter mismatch: got %u, expected %u\n",
                     dmaIdx, pgpIndex, m_batchStart + last);

    sawDisable = th->service() == TransitionId::Disable;

    PGPEvent*  event  = &m_pool.pgpEvents()[pgpIndex];
    DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
    buffer->size = size;                       //   "
    buffer->index = dsc->index;                //   "
    event->mask |= (1 << lane);

    // Allocate a pebble buffer once the event is built
    auto counter       = m_pool.allocate(); // This can block
    auto pebbleIndex   = counter & (m_pool.nbuffers() - 1);
    event->pebbleIndex = pebbleIndex;

    // Make a new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    auto dgram = new(m_pool.pebble()[pebbleIndex]) EbDgram(*th, m_det.nodeId, m_para.rogMask);

    // @todo: Process the data to extract TEB input and calibrate.  Also reduce/compress?
    //if (th->service() == TransitionId::L1Accept)
    //  this->event(*th, event);
    //else if (th->service() == TransitionId::SlowUpdate)
    //  this->slowUpdate(*th);

    last        += nDmaRet;
    m_batchSize += nDmaRet;
    full = m_batchSize == 4;           // @todo: arbitrary

    logging::debug("*** %d nDmaRet %d, last %u, size %u, full %d, sawDisable %d\n",
                   dmaIdx, nDmaRet, last, m_batchSize.load(), full, sawDisable);

    if (full || sawDisable) {
      // Ensure PgpReader::handle() doesn't complain about jumps
      m_lastEvtCtr = th->evtCounter;

      // Queue the batch to the Collector
      collectorGpuQueue.push({m_batchStart, m_batchSize});

      // Reset to the beginning of the batch
      full = false;
      last = 0;
      m_batchStart = pgpIndex + 1;
      m_batchSize  = 0;
    }
  }

  // Clean up
  free(hostWriteBuf);

  logging::debug("Returning from reader[%d]\n", dmaIdx);
}
