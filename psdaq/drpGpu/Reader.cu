#include "Reader.hh"

#include "Detector.hh"
#include "drp/spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <thread>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;

struct rdr_domain{ static constexpr char const* name{"Reader"}; };
using rdr_scoped_range = nvtx3::scoped_range_in<rdr_domain>;


Reader::Reader(const Parameters&                  para,
               MemPoolGpu&                        pool,
               Detector&                          det,
               size_t                             trgPrimitiveSize,
               const cudaExecutionContext_t&      green_ctx,
               const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool       (pool),
  m_det        (det),
  m_ctx        (green_ctx),
  m_terminate_d(terminate_d),
  m_para       (para)
{
  // Set up pebble index allocator for managing pebble buffers
  m_pebbleQueue.h = new RingIndexHtoD(m_pool.nbuffers());
  chkError(cudaMalloc(&m_pebbleQueue.d,                  sizeof(*m_pebbleQueue.d)));
  chkError(cudaMemcpy( m_pebbleQueue.d, m_pebbleQueue.h, sizeof(*m_pebbleQueue.d), cudaMemcpyDefault));

  // Set up buffer index allocator for DMA to Collector comms
  m_readerQueue.h = new RingIndexDtoD(m_pool.nbuffers());
  chkError(cudaMalloc(&m_readerQueue.d,                  sizeof(*m_readerQueue.d)));
  chkError(cudaMemcpy( m_readerQueue.d, m_readerQueue.h, sizeof(*m_readerQueue.d), cudaMemcpyDefault));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo};
  logging::debug("Reader stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Allocate a stream at lowest priority so that higher priority can
  // be given to downstream stages that help drain the system
  //chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));
  chkFatal(cudaExecutionCtxStreamCreate(&m_stream, green_ctx, cudaStreamNonBlocking, prio));

  const auto panel = m_pool.panel();
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    printf("*** Reader: dmaBufIdx %u, hwWrtPtr %p, hwWrtStart %p\n",
           i, (void*)(panel->dmaBuffers[i]), (uint8_t*)panel->fpgaRegs.d + panel->coreRegs.freeListOffset(i));
  }

  // Prepare buffers visible to the host for receiving headers
  const size_t bufSz = sizeof(DmaDsc) + sizeof(TimingHeader) + trgPrimitiveSize;
  m_pool.createHostBuffers(bufSz);

  // Set up a state variable
  chkError(cudaMalloc(&m_state_d,    sizeof(*m_state_d)));
  chkError(cudaMemset( m_state_d, 0, sizeof(*m_state_d)));

  // Prepare a metric for tracking kernel state
  chkError(cudaHostAlloc(&m_metrics.state.h, sizeof(*m_metrics.state.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.state.d, m_metrics.state.h, 0));
  *m_metrics.state.h = 0;

  // Prepare counter metrics for tracking execution progress
  chkError(cudaHostAlloc(&m_metrics.pblWtCtr.h, sizeof(*m_metrics.pblWtCtr.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.pblWtCtr.d, m_metrics.pblWtCtr.h, 0));
  *m_metrics.pblWtCtr.h = 0;
  chkError(cudaHostAlloc(&m_metrics.dmaWtCtr.h, sizeof(*m_metrics.dmaWtCtr.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.dmaWtCtr.d, m_metrics.dmaWtCtr.h, 0));
  *m_metrics.dmaWtCtr.h = 0;
  chkError(cudaHostAlloc(&m_metrics.fwdWtCtr.h, sizeof(*m_metrics.fwdWtCtr.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.fwdWtCtr.d, m_metrics.fwdWtCtr.h, 0));
  *m_metrics.fwdWtCtr.h = 0;

  // Prepare the CUDA graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Reader graph");
    abort();
  }
}

Reader::~Reader()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  if (m_metrics.pblWtCtr.h) {
    chkError(cudaFreeHost(m_metrics.pblWtCtr.h));
    m_metrics.pblWtCtr.h = nullptr;
    m_metrics.pblWtCtr.d = nullptr;
  }
  if (m_metrics.dmaWtCtr.h) {
    chkError(cudaFreeHost(m_metrics.dmaWtCtr.h));
    m_metrics.dmaWtCtr.h = nullptr;
    m_metrics.dmaWtCtr.d = nullptr;
  }
  if (m_metrics.fwdWtCtr.h) {
    chkError(cudaFreeHost(m_metrics.fwdWtCtr.h));
    m_metrics.fwdWtCtr.h = nullptr;
    m_metrics.fwdWtCtr.d = nullptr;
  }

  if (m_metrics.state.h) {
    chkError(cudaFreeHost(m_metrics.state.h));
    m_metrics.state.h = nullptr;
    m_metrics.state.d = nullptr;
  }

  if (m_state_d)  chkError(cudaFree(m_state_d));
  m_state_d = nullptr;

  m_pool.destroyHostBuffers();

  chkError(cudaStreamDestroy(m_stream));

  if (m_readerQueue.d)  chkError(cudaFree(m_readerQueue.d));
  if (m_readerQueue.h)  delete m_readerQueue.h;
  if (m_pebbleQueue.d)  chkError(cudaFree(m_pebbleQueue.d));
  if (m_pebbleQueue.h)  delete m_pebbleQueue.h;
}

int Reader::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                         std::map<std::string, std::string>&   labels)
{
  *m_metrics.state.h = 0;
  *m_metrics.pblWtCtr.h = 0;
  *m_metrics.dmaWtCtr.h = 0;
  *m_metrics.fwdWtCtr.h = 0;
  exporter->add("DRP_readerState", labels, MetricType::Gauge,   [&](){ return m_metrics.state.h    ? *m_metrics.state.h    : 0; });

  exporter->add("DRP_pblWtCtr", labels, MetricType::Counter, [&](){ return m_metrics.pblWtCtr.h ? *m_metrics.pblWtCtr.h : 0; });
  exporter->add("DRP_dmaWtCtr", labels, MetricType::Counter, [&](){ return m_metrics.dmaWtCtr.h ? *m_metrics.dmaWtCtr.h : 0; });
  exporter->add("DRP_rdrFwd",   labels, MetricType::Counter, [&](){ return m_metrics.fwdWtCtr.h ? *m_metrics.fwdWtCtr.h : 0; });

  exporter->add("DRP_pblQueOcc", labels, MetricType::Gauge,   [&](){ return m_pebbleQueue.h->occupancy(); });
  exporter->add("DRP_rdrQueOcc", labels, MetricType::Gauge,   [&](){ return m_readerQueue.h->occupancy(); });

  return 0;
}

int Reader::_setupGraph()
{
  // Generate the graph
  logging::debug("Recording Reader graph");
  auto graph = _recordGraph();
  if (graph == 0) {
    return -1;
  }

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reader graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reader graph...");
  if (chkError(cudaGraphUpload(m_graphExec, m_stream), "Reader graph upload failed")) {
    return -1;
  }

  return 0;
}

static __device__
void _calibrate(float*        const        __restrict__ calib,
                uint16_t      const* const __restrict__ raw,
                unsigned      const                     nElements,
                unsigned      const                     rangeOffset,
                unsigned      const                     rangeBits,
                float         const* const __restrict__ pedArray,
                float         const* const __restrict__ gainArray,
                float         const* const __restrict__ ref)
{
  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;

  auto const rangeMask{(1 << rangeBits) - 1};
  auto const dataMask {(1 << rangeOffset) - 1};
  for (auto i = tid; i < nElements; i += stride) {
    auto const              range = (raw[i] >> rangeOffset) & rangeMask;
    auto const __restrict__ peds  = &pedArray [range * nElements];
    auto const __restrict__ gains = &gainArray[range * nElements];
    auto const              data  = raw[i] & dataMask;
    calib[i] = (float(data) - peds[i]) * gains[i];

    //if (i < 4) {
    //  printf("### Reader: tid %u, i %u: raw %p: %04x, dat %u, rng %u, ped %f, gn %f, cal %f, ref %p: %f\n",
    //         tid, i, &raw[i], raw[i], data, range, peds[i], gains[i], calib[i], &ref[i], ref ? ref[i] : 0.f);
    //}
    //if (ref && (calib[i] != ref[i])) {
    //  printf("### Reader: blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n", blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
    //}
  }
}

static __device__ unsigned lDmaBufferIdx = 0;
static __device__ unsigned lDmaBufferNxt = 0;
static __device__ unsigned lPebbleIdx    = 0;

// Wait for the DMA size word to become non-zero
static __global__
void _waitForDMA(unsigned* const               __restrict__ state,
                 uint8_t*  const               __restrict__ wrEnReg,
                 uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
                 size_t    const                            dmaCount,
                 RingIndexHtoD*   const        __restrict__ pebbleQueue,
                 RingIndexDtoD*   const        __restrict__ readerQueue,
                 uint64_t* const               __restrict__ stateMon,
                 uint64_t* const               __restrict__ pblWtCtr,
                 uint64_t* const               __restrict__ dmaWtCtr)
{
  //*stateMon = 1;

  if (*state == 0) {
    // Allocate the index of the next set of intermediate buffers to be used
    unsigned ns{8};
    while (!pebbleQueue->pop(&lPebbleIdx)) { // This blocks when no buffers available
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 2;
        return;
      }
    }
    //printf("### Reader: got pblIdx %u\n", lPebbleIdx);
    *state = 1;
    //*stateMon = 3;
    //++(*pblWtCtr);
  }
  if (*state == 1) {
    auto dmaBufIdx = lDmaBufferIdx;     // All threads load DMA idx into a register from global memory
    auto const __restrict__ dmaBufs = &dmaBuffers[0];

    // Wait for data to be DMAed into the GPU
    const volatile uint32_t* const __restrict__ mem = (uint32_t*)(dmaBufs[dmaBufIdx] + 4);
    unsigned ns{8};
    //printf("### Reader::handleDMA: Wait for dmaBufs[%u] %p\n", dmaBufIdx, mem);
    while (*mem == 0) {                 // Wait for DMA completion
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 4;
        return;
      }
    }
    *state = 2;
    //*stateMon = 5;
    //++(*dmaWtCtr);
    //printf("### Reader: dma[%u] %p: sz %u\n", dmaBufIdx, mem, *mem);

    auto next = (dmaBufIdx + 1) & (dmaCount - 1);      // Prepare for the next DMA buffer
    *(volatile uint32_t*)(dmaBufs[next] + 4) = 0;      // Clear the handshake space of the next DMA buffer
    *(volatile uint8_t*)(wrEnReg + next * 4) = 1;      // Enable the DMA on this dataDev
    lDmaBufferNxt = next;                              // Update global memory
    //printf("### Reader: next %lu, hand shake %p, next write enable %p\n", next, dmaBufs[next] + 4, wrEnReg + next * 4);
  }
}

// This copies the DmaDsc and TimingHeader into a host-visible buffer
static __global__
void _event(unsigned* const               __restrict__ state,
            uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
            size_t    const                            frameSize,
            uint32_t* const               __restrict__ hdrBuffers,    // [nBuffers * hdrBufsCnt]
            size_t    const                            hdrBufsCnt,
            float*    const               __restrict__ calibBuffers,  // [nBuffers * calibBufsCnt]
            size_t    const                            calibBufsCnt,
            float     const* const        __restrict__ pedArray,
            float     const* const        __restrict__ gainArray,
            unsigned  const                            rangeOffset,
            unsigned  const                            rangeBits,
            float     const* const        __restrict__ refBuffers,
            unsigned  const                            refBufCnt,
            uint64_t* const               __restrict__ stateMon)
{
  if (*state == 2) {
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const __restrict__ dmaBufs = &dmaBuffers[0];

    auto dmaBufIdx = lDmaBufferIdx; // All threads load DMA idx into a register from global memory
    auto pblBufIdx = lPebbleIdx;    // All threads load pebble idx into a register from global memory

    // Save the DMA descriptor and TimingHeader in pinned memory
    //if (tid == 0)  printf("### Reader: dmaIdx %3u, dmaBuf %p, pblIdx %4u\n", dmaBufIdx, dmaBufs, pblBufIdx);
    auto const __restrict__ in  = (uint32_t*)dmaBufs[dmaBufIdx];
    //if (threadIdx.x == 0)  printf("### Reader: blk %3d, dmaIdx %2u, pblIdx %4u, in %p, in[1] %u, in[9:8] %08x, %08x\n",
    //                              blockIdx.x, dmaBufIdx, pblBufIdx, in, in[1], in[9], in[8]);
    //if (tid == 0)  printf("### Reader: hdrBufs %p\n", hdrBuffers);
    auto const __restrict__ hdr = hdrBuffers + pblBufIdx * hdrBufsCnt;
    //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader: blk %3d, ob %p, pblIdx %u, hdrBufsCnt %lu, hdr %p\n", blockIdx.x, hdrBuffers, pblBufIdx, hdrBufsCnt, hdr);
    constexpr auto nDscWds = sizeof(DmaDsc)/sizeof(*in);
    constexpr auto nHdrWds = sizeof(TimingHeader)/sizeof(*in);
    constexpr auto nLdrWds = nDscWds + nHdrWds;
    const     auto nFrmWds = frameSize/sizeof(*in);
    //if (tid == 0)  printf("### Reader: nDscWds %lu, nLdrWds %lu\n", nDscWds, nLdrWds);
    if      (tid < nDscWds)  { hdr[tid] = in[tid]; } //printf("### Reader: tid %d, hdr %08x\n", tid, hdr[tid]); }
    else if (tid < nLdrWds)  { hdr[tid] = in[nFrmWds + tid - nDscWds]; } //printf("### Reader: tid %d, i %lu, hdr %08x\n", tid, nFrmWds + tid - nDscWds, hdr[tid]); }
    //else if (tid < nLdrWds+2) { printf("### Reader: tid %d, i %lu, %04x %04x\n", tid, nFrmWds + tid - nDscWds, in[nFrmWds + tid - nDscWds] & 0xffff, (in[nFrmWds + tid - nDscWds]>>16)&0xffff); }

    // Calibrate
    //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader: blk %3d, in[1] %u, th sz %lu\n", blockIdx.x, in[1], sizeof(TimingHeader));
    auto const payloadSz = in[1] - sizeof(TimingHeader);
    //if (tid == 0)  printf("### Reader: dmaSz %u, pyldSz %ld\n", in[1], payloadSz);
    if (payloadSz > 0) { // Calibrate only when there's a payload
      auto const __restrict__ raw = (uint16_t*)&in[nFrmWds + nHdrWds];
      //if (tid == 0)  printf("### Reader: raw %p\n", raw);
      auto const __restrict__ out = &calibBuffers[pblBufIdx * calibBufsCnt];
      //if (tid == 0)  printf("### Reader: out %p\n", out);
      auto const payloadCnt = payloadSz/sizeof(*raw);
      //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu\n", payloadCnt, calibBufsCnt);
      auto const elementCnt = payloadCnt > calibBufsCnt ? calibBufsCnt : payloadCnt; // @todo: Alert to truncation
      //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu, cnt %lu\n",
      //                      payloadCnt, calibBufsCnt, elementCnt);
      auto const __restrict__ ref = refBuffers ? &refBuffers[(pblBufIdx % refBufCnt) * calibBufsCnt] : (float*)0;
      //if (tid == 0)  printf("### Reader: idx %u, refBuffers %p, ref %p\n", pblBufIdx%refBufCnt, refBuffers, ref);

      //auto const stride  = blockDim.x * gridDim.x;
      //if (tid == 0)  printf("### calibrate: nElements %lu, stride %u, idx %u, calib %p:%p\n",
      //                       elementCnt, stride, pblBufIdx, &out[0],&out[elementCnt]);

      //if (tid == 0) *stateMon = 11;
      _calibrate(out, raw, elementCnt, rangeOffset, rangeBits, pedArray, gainArray, ref);
      //if (tid == 0) *stateMon = 12;
    }

    // Advance to the next state
    // State variable is likely set before the last thread is done, but
    // the next kernel won't check it before all this kernel completes
    if (tid == 0) *state = 3;
  }
}

// This will re-launch the current graph
static __global__
void _readerLoop(unsigned*      const __restrict__  state,
                 RingIndexDtoD* const __restrict__  readerQueue,
                 cuda::std::atomic<unsigned> const& terminate,
                 uint64_t*      const __restrict__  stateMon,
                 uint64_t*      const __restrict__  evtCtr)
{
  if (*state == 3) {
    //*stateMon = 13;
    bool rc;
    unsigned ns{8};
    while ( (rc = !readerQueue->push(lPebbleIdx)) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 14;
        break;
      }
    }
    if (!rc) {
      //printf("### Reader: pushed pblIdx %u\n", lPebbleIdx);
      lDmaBufferIdx = lDmaBufferNxt;
      *state = 0;
      //*stateMon = 15;
      //++(*evtCtr);
    }
  }

  // Relaunch the graph
  //printf("### Reader: 2 relaunch\n");
  if (!terminate.load(cuda::std::memory_order_acquire)) {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
}

/******************************************************************************
 * Records a CUDA graph for later instantiation and execution.
 * The nodes within a CUDA graph define the execution steps of what amounts to
 * a "command buffer", in traditional graphics terms.  Edges between nodes on
 * the graph define dependencies.  The execution flow of the GPU DRP
 * application can be accurately described using the graph structure as defined
 * by the CUDA graph API.  Normally, CUDA API calls that run on the GPU (i.e.
 * cuStreamWriteXXX) are converted into an internal representation and inserted
 * into a command buffer within the CUDA driver.  The sync functions can then
 * be used to describe dependencies between steps, however this involves the
 * host and thus introduces latency between steps.  In the case of CUDA graphs,
 * we can avoid host involvement completely and simply give the GPU a list of
 * instructions to execute.  We can even tell the GPU to launch new graphs on
 * its own, if we wanted to cut host involvement out entirely.
 ******************************************************************************/
cudaGraph_t Reader::_recordGraph()
{
  rdr_scoped_range r{/*"Reader::_recordGraph"*/}; // Expose function name via NVTX

  auto       panel          = m_pool.panel();
  auto const wrEnReg_d      = (uint8_t*)panel->fpgaRegs.d + panel->coreRegs.freeListOffset(0);
  auto const dmaBuffers_d   = panel->dmaBuffers_d;
  auto const dmaCount       = m_pool.dmaCount();
  auto const frameSize      = panel->coreRegs.dmaDataBytes();
  auto const hostWrtBufs_d  = m_pool.hostWrtBufs_d();
  auto const hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs_d);
  auto const calibBuffers_d = m_pool.calibBuffers_d();
  auto const calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers_d);
  auto const rangeOffset    = m_det.rangeOffset();
  auto const rangeBits      = m_det.rangeBits();
  auto const refBuffers_d   = m_det.referenceBuffers();
  auto const refBufCnt      = m_det.referenceBufCnt();
  auto const pedArray_d     = m_det.pedestals_d();
  auto const gainArray_d    = m_det.gains_d();

  // Determine how many processing resources to reserve for the Reader kernel
  // @todo: The maybe should be done in PgpDetector in conjunction with the other components
  cudaDeviceProp prop;
  chkError(cudaGetDeviceProperties(&prop, 0));
  const auto tpSM{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;
  switch (tpSM) {
    case 1536:  nSMs = 4;  break;
    case 2048:  nSMs = 2;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpSM);
      abort();
  };
//  cudaDevResource smResources = {};
//  chkError(cudaExecutionCtxGetDevResource(m_ctx,                   // Reader's context
//                                          &smResources,            // Device resource to populate
//                                          cudaDevResourceTypeSm)); // Resource type
//  const auto nSMs{smResources.sm.smCount};
  // Slightly better times seem to be achieved when nPixels/stride is an integer
  // Adjusting nBlocks for this might lead to a partially used SM, but aim for
  // maximum occupancy of the SMs
  const auto maxBpSM{prop.maxBlocksPerMultiProcessor};
  auto nThreads{tpSM/maxBpSM}; //{32}; // @todo: Move to green contexts for improved robustness
  auto nBlocks{nSMs*maxBpSM}; //{63}; //{189}; //{nSMs * tpSM / nThreads}; // {189};
  auto stride{nBlocks * nThreads};
  printf("*** Reader: blocks %u * threads %u = %u threads\n", nBlocks, nThreads, stride);

  logging::info("GPU threads per SM: %d, total threads: %u, SMs %.1f, elements per thread: %.1f\n",
                tpSM, stride, float(stride) / tpSM, float(calibBufsCnt) / stride);

//  CUcontext ctx;
//  chkError(cuCtxFromGreenCtx(&ctx, m_ctx));
//  chkError(cuCtxSetCurrent(ctx));

  if (chkError(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal),
               "Stream begin-capture failed")) {
    return 0;
  }

  _waitForDMA<<<1, 1, 1, m_stream>>>(m_state_d,
                                     wrEnReg_d,
                                     dmaBuffers_d,
                                     dmaCount,
                                     m_pebbleQueue.d,
                                     m_readerQueue.d,
                                     m_metrics.state.d,
                                     m_metrics.pblWtCtr.d,
                                     m_metrics.dmaWtCtr.d);
  chkError(cudaGetLastError(), "Launch of _waitForDMA kernel failed");

  // Copy the DMA descriptor and the timing header to host-visible pinned memory buffers
  // Calibrate the raw data from the DMA buffers into the calibrated data buffers
  //constexpr auto iPayload { (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(uint32_t) };
  _event<<<nBlocks, nThreads, 0, m_stream>>>(m_state_d,
                                             dmaBuffers_d,
                                             frameSize,
                                             hostWrtBufs_d,
                                             hostWrtBufsCnt,
                                             calibBuffers_d,
                                             calibBufsCnt,
                                             pedArray_d,
                                             gainArray_d,
                                             rangeOffset,
                                             rangeBits,
                                             refBuffers_d,
                                             refBufCnt,
                                             m_metrics.state.d);
  chkError(cudaGetLastError(), "Launch of _event kernel failed");

  // Publish the current head index and re-launch
  _readerLoop<<<1, 1, 0, m_stream>>>(m_state_d,
                                     m_readerQueue.d,
                                     m_terminate_d,
                                     m_metrics.state.d,
                                     m_metrics.fwdWtCtr.d);
  chkError(cudaGetLastError(), "Launch of _readerLoop kernel failed");

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(m_stream, &graph), "Stream end-capture failed")) {
    return 0;
  }

  printf("*** Reader: Returning graph\n");
  return graph;
}

void Reader::start()
{
  logging::info("Reader is starting");

  auto const panel = m_pool.panel();

#ifdef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA
  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    auto rc = gpuSetWriteEn(panel->datadev.fd(), dmaIdx);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
      abort();
    }
  }
#endif // HOST_REARMS_DMA

  // Enable a DMA for buffer 0 only
  unsigned dmaBufIdx{0};
  // Clear handshake space on the GPU side (A.K.A. "GPU's doorbell")
  const auto dmaBufs = panel->dmaBuffers[dmaBufIdx];
  chkError(cudaMemsetAsync(dmaBufs + 4, 0, sizeof(uint32_t), m_stream));
  printf("*** Reader: dmaBufIdx %u, dmaBuffer %p\n", dmaBufIdx, (void*)dmaBufs);

#ifndef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA to trigger the write
  panel->coreRegs.returnFreeListIndex(dmaBufIdx);
#endif // HOST_REARMS_DMA

  // Ensure that the DMA round-robin index starts with buffer 0
  dmaIdxReset(panel->coreRegs);

  // Launch the Reader graph
  printf("*** Reader: Launching graph\n");
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}
