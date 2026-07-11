#include "Reader.hh"

#include "Detector.hh"
#include "drp/spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"
#include "psdaq/trigger/src/TriggerPrimitive.hh"

#include <thread>

#if 0                                   // Set to 1 to enable device metrics  These affect performance.
#define DBG(stmt) stmt
#else
#define DBG(stmt)
#endif

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
  // Establish the number of Reader graphs to use
  if (para.kwargs.find("nReaders") != para.kwargs.end())
    m_nReaders = std::stoul(para.kwargs.at("nReaders"));
  else
    m_nReaders = 1;
  if ((m_nReaders == 0) || (m_nReaders * (m_pool.dmaCount()/m_nReaders)) != m_pool.dmaCount()) {
    logging::critical("The number of Readers (%u) must divide into the number of DMA buffers (%u) evenly",
                      m_nReaders, m_pool.dmaCount());
    abort();
  }
  if (m_nReaders & (m_nReaders-1)) { // GPU divides by non-powers-of-2 are expensive
    logging::critical("The number of DMA reader streams must be a power of 2; got %u",
                      m_nReaders);
    abort();
  }

  logging::info("Running with %u Reader streams", m_nReaders);

  // Create DMA buffer index counters for each Reader
  m_dmaBufferIdxes.resize(m_nReaders);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaMalloc(&m_dmaBufferIdxes[i],     sizeof(*m_dmaBufferIdxes[i])));
    chkError(cudaMemcpy( m_dmaBufferIdxes[i], &i, sizeof(*m_dmaBufferIdxes[i]), cudaMemcpyDefault));
  }

  // Create pebble buffer index counters for each Reader
  m_pebbleIdxes.resize(m_nReaders);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaMalloc(&m_pebbleIdxes[i],    sizeof(*m_pebbleIdxes[i])));
    chkError(cudaMemset( m_pebbleIdxes[i], 0, sizeof(*m_pebbleIdxes[i])));
  }

  // Set up pebble index allocator for managing pebble buffers
  constexpr bool initFull{false};
  m_pebbleQueue.h = new RingIndexHtoD(m_pool.nbuffers(), initFull);
  chkError(cudaMalloc(&m_pebbleQueue.d,                  sizeof(*m_pebbleQueue.d)));
  chkError(cudaMemcpy( m_pebbleQueue.d, m_pebbleQueue.h, sizeof(*m_pebbleQueue.d), cudaMemcpyDefault));

  // Set up buffer index allocator for DMA to TrgInpGen comms
  m_readerQueues.resize(m_nReaders);
  auto nBuffers = m_pool.nbuffers() / m_nReaders;
  for (unsigned i = 0; i < m_nReaders; ++i) {
    m_readerQueues[i].h = new RingIndexDtoD(nBuffers);
    chkError(cudaMalloc(&m_readerQueues[i].d,                      sizeof(*m_readerQueues[i].d)));
    chkError(cudaMemcpy( m_readerQueues[i].d, m_readerQueues[i].h, sizeof(*m_readerQueues[i].d), cudaMemcpyDefault));
  }

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo};
  logging::debug("Reader stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Allocate a stream at lowest priority so that higher priority can
  // be given to downstream stages that help drain the system
  m_streams.resize(m_nReaders);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    //chkFatal(cudaStreamCreateWithPriority(&m_streams[i], cudaStreamNonBlocking, prio));
    chkFatal(cudaExecutionCtxStreamCreate(&m_streams[i], green_ctx, cudaStreamNonBlocking, prio));
  }

  // Prepare buffers visible to the host for receiving headers
  const size_t bufSz = sizeof(DmaDsc) + sizeof(TimingHeader) + trgPrimitiveSize;
  m_pool.createHostBuffers(bufSz);

  // Set up a state variable
  m_states_d.resize(m_nReaders);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaMalloc(&m_states_d[i],    sizeof(*m_states_d[i])));
    chkError(cudaMemset( m_states_d[i], 0, sizeof(*m_states_d[i])));
  }

  // Prepare a metric for tracking kernel state
  m_metrics.states.resize(m_nReaders, nullptr);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaHostAlloc(&m_metrics.states[i], sizeof(*m_metrics.states[i]), cudaHostAllocDefault));
    *m_metrics.states[i] = 0;
  }

  // Prepare counter metrics for tracking execution progress
  m_metrics.pblWtCtrs.resize(m_nReaders, nullptr);
  m_metrics.dmaWtCtrs.resize(m_nReaders, nullptr);
  m_metrics.fwdWtCtrs.resize(m_nReaders, nullptr);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaHostAlloc(&m_metrics.pblWtCtrs[i], sizeof(*m_metrics.pblWtCtrs[i]), cudaHostAllocDefault));
    *m_metrics.pblWtCtrs[i] = 0;
    chkError(cudaHostAlloc(&m_metrics.dmaWtCtrs[i], sizeof(*m_metrics.dmaWtCtrs[i]), cudaHostAllocDefault));
    *m_metrics.dmaWtCtrs[i] = 0;
    chkError(cudaHostAlloc(&m_metrics.fwdWtCtrs[i], sizeof(*m_metrics.fwdWtCtrs[i]), cudaHostAllocDefault));
    *m_metrics.fwdWtCtrs[i] = 0;
  }

  // Prepare the CUDA graph
  m_graphExecs.resize(m_nReaders);
  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (_setupGraph(i)) {
      logging::critical("Failed to set up Reader[%u]'s graph", i);
      abort();
    }
  }
}

Reader::~Reader()
{
  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaGraphExecDestroy(m_graphExecs[i]));
  }
  m_graphExecs.clear();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_metrics.pblWtCtrs[i]) {
      chkError(cudaFreeHost(m_metrics.pblWtCtrs[i]));
      m_metrics.pblWtCtrs[i] = nullptr;
    }
    if (m_metrics.dmaWtCtrs[i]) {
      chkError(cudaFreeHost(m_metrics.dmaWtCtrs[i]));
      m_metrics.dmaWtCtrs[i] = nullptr;
    }
    if (m_metrics.fwdWtCtrs[i]) {
      chkError(cudaFreeHost(m_metrics.fwdWtCtrs[i]));
      m_metrics.fwdWtCtrs[i] = nullptr;
    }
  }
  m_metrics.pblWtCtrs.clear();
  m_metrics.dmaWtCtrs.clear();
  m_metrics.fwdWtCtrs.clear();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_metrics.states[i]) {
      chkError(cudaFreeHost(m_metrics.states[i]));
      m_metrics.states[i] = nullptr;
    }
  }
  m_metrics.states.clear();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_states_d[i])  chkError(cudaFree(m_states_d[i]));
    m_states_d[i] = nullptr;
  }
  m_states_d.clear();

  m_pool.destroyHostBuffers();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    chkError(cudaStreamDestroy(m_streams[i]));
  }
  m_streams.clear();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_readerQueues[i].d)  chkError(cudaFree(m_readerQueues[i].d));
    if (m_readerQueues[i].h)  delete m_readerQueues[i].h;
  }
  m_readerQueues.clear();
  if (m_pebbleQueue.d)  chkError(cudaFree(m_pebbleQueue.d));
  if (m_pebbleQueue.h)  delete m_pebbleQueue.h;

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_pebbleIdxes[i])  chkError(cudaFree(m_pebbleIdxes[i]));
  }
  m_pebbleIdxes.clear();

  for (unsigned i = 0; i < m_nReaders; ++i) {
    if (m_dmaBufferIdxes[i])  chkError(cudaFree(m_dmaBufferIdxes[i]));
  }
  m_dmaBufferIdxes.clear();
}

int Reader::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                         std::map<std::string, std::string>&   labels)
{
  for (unsigned i = 0; i < m_nReaders; ++i) {
    *m_metrics.states[i] = 0;
    auto rdr = std::to_string(i);
    exporter->add("DRP_readerState"+rdr, labels, MetricType::Gauge, [&, i](){ return m_metrics.states[i] ? *m_metrics.states[i] : 0; });
  }

  for (unsigned i = 0; i < m_nReaders; ++i) {
    *m_metrics.pblWtCtrs[i] = 0;
    *m_metrics.dmaWtCtrs[i] = 0;
    *m_metrics.fwdWtCtrs[i] = 0;
    auto rdr = std::to_string(i);
    exporter->add("DRP_pblWtCtr"+rdr, labels, MetricType::Counter, [&, i](){ return m_metrics.pblWtCtrs[i] ? *m_metrics.pblWtCtrs[i] : 0; });
    exporter->add("DRP_dmaWtCtr"+rdr, labels, MetricType::Counter, [&, i](){ return m_metrics.dmaWtCtrs[i] ? *m_metrics.dmaWtCtrs[i] : 0; });
    exporter->add("DRP_rdrFwd"+rdr,   labels, MetricType::Counter, [&, i](){ return m_metrics.fwdWtCtrs[i] ? *m_metrics.fwdWtCtrs[i] : 0; });
  }

  exporter->add("DRP_pblQueOcc", labels, MetricType::Gauge, [&](){ return m_pebbleQueue.h->occupancy(); });
  for (unsigned i = 0; i < m_nReaders; ++i) {
    auto rdr = std::to_string(i);
    exporter->add("DRP_rdrQueOcc"+rdr, labels, MetricType::Gauge, [&, i](){ return m_readerQueues[i].h->occupancy(); });
  }

  return 0;
}

int Reader::_setupGraph(unsigned reader)
{
  // Generate the graph
  logging::debug("Recording Reader graph %u", reader);
  auto graph = _recordGraph(reader);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExecs[reader], graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reader graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reader[%u]'s graph...", reader);
  if (chkError(cudaGraphUpload(m_graphExecs[reader], m_streams[reader]), "Reader graph upload failed")) {
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

// Wait for the DMA size word to become non-zero
static __global__
void _waitForDMA(unsigned  const                            reader,
                 unsigned  const                            nReaders,
                 unsigned* const               __restrict__ state,
                 unsigned* const               __restrict__ dmaBufferIdx,
                 unsigned* const               __restrict__ pebbleIdx,
                 uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
                 RingIndexHtoD*   const        __restrict__ pebbleQueue,
                 uint64_t* const               __restrict__ stateMon,
                 uint64_t* const               __restrict__ pblWtCtr,
                 uint64_t* const               __restrict__ dmaWtCtr)
{
  auto lclState{*state};
  if (lclState == 0) {
    DBG(*stateMon = 1;)
    // Wait for data to be DMAed into the GPU
    const volatile uint32_t* const __restrict__ mem = (uint32_t*)(dmaBuffers[*dmaBufferIdx] + 4);
    unsigned ns{8};
    //printf("### Reader[%u]: Wait for dmaBuffers[%u] %p\n", reader, *dmaBufferIdx, mem);
    while (*mem == 0) {                 // Wait for DMA completion
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        DBG(*stateMon = 2;)
        return;
      }
    }
    //printf("### Reader[%u]: dma[%u] %p: sz %u\n", reader, *dmaBufferIdx, mem, *mem);
    lclState = 1;
    *state = lclState;
    DBG(*stateMon = 3; ++(*dmaWtCtr);)
  }
  if (lclState == 1) {
    // Allocate the index of the next set of intermediate buffers to be used
    unsigned ns{8};
    unsigned idx;
    while (!(((pebbleQueue->tail() & (nReaders-1)) == reader) && // Preserve allocation order by stream
             pebbleQueue->pop(&idx))) { // This blocks when no buffers available
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        DBG(*stateMon = 4;)
        return;
      }
    }
    *pebbleIdx = idx;
    //printf("### Reader[%u]: got pblIdx %u, hd %u, tl %u, occ %u\n", reader, idx,
    //       pebbleQueue->head(), pebbleQueue->tail(), pebbleQueue->occupancy());
    *state = 2;
    DBG(*stateMon = 5; ++(*pblWtCtr);)
  }
}

// This copies the DmaDsc and TimingHeader into a host-visible buffer
static __global__
void _event(unsigned  const                            reader,
            unsigned* const               __restrict__ state,
            unsigned* const               __restrict__ dmaBufferIdx,
            unsigned* const               __restrict__ pebbleIdx,
            uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
            size_t    const                            frameSize,
            uint32_t* const               __restrict__ hdrBuffers,    // [nBuffers * hdrBufsCnt]
            size_t    const                            hdrBufsCnt,
            float*    const               __restrict__ calibBuffers,  // [nBuffers * calibBufsCnt]
            size_t    const                            calibBufsCnt,
            float     const* const        __restrict__ pedArray,
            float     const* const        __restrict__ gainArray,
            //auto      const                            calibFn,       // Not working
            unsigned  const                            rangeOffset,
            unsigned  const                            rangeBits,
            float     const* const        __restrict__ refBuffers,
            unsigned  const                            refBufCnt,
            uint64_t* const               __restrict__ stateMon)
{
  if (*state == 2) {
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const __restrict__ dmaBufs = &dmaBuffers[0];

    auto const dmaBufIdx{*dmaBufferIdx}; // All threads load DMA idx into a register from global memory
    auto const pblBufIdx{*pebbleIdx};    // All threads load pebble idx into a register from global memory

    // Save the DMA descriptor and TimingHeader in pinned memory
    //if (tid == 0)  printf("### Reader[%u]: dmaIdx %3u, dmaBuf %p, pblIdx %4u\n", reader, dmaBufIdx, dmaBufs, pblBufIdx);
    auto const __restrict__ in  = (uint32_t*)dmaBufs[dmaBufIdx];
    //if (threadIdx.x == 0)  printf("### Reader[%u]: blk %3d, dmaIdx %2u, pblIdx %4u, in %p, in[1] %u, in[9:8] %08x, %08x\n",
    //                              reader, blockIdx.x, dmaBufIdx, pblBufIdx, in, in[1], in[9], in[8]);
    //if (tid == 0)  printf("### Reader[%u]: hdrBufs %p\n", reader, hdrBuffers);
    auto const __restrict__ hdr = hdrBuffers + pblBufIdx * hdrBufsCnt;
    //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader[%u]: blk %3d, ob %p, pblIdx %u, hdrBufsCnt %lu, hdr %p\n", reader, blockIdx.x, hdrBuffers, pblBufIdx, hdrBufsCnt, hdr);
    constexpr auto nDscWds = sizeof(DmaDsc)/sizeof(*in);
    constexpr auto nHdrWds = sizeof(TimingHeader)/sizeof(*in);
    constexpr auto nLdrWds = nDscWds + nHdrWds;
    const     auto nFrmWds = frameSize/sizeof(*in);
    //if (tid == 0)  printf("### Reader[%u]: nDscWds %lu, nLdrWds %lu\n", reader, nDscWds, nLdrWds);
    if      (tid < nDscWds)  { hdr[tid] = in[tid]; } //printf("### Reader[%u]: tid %d, hdr %08x\n", reader, tid, hdr[tid]); }
    else if (tid < nLdrWds)  { hdr[tid] = in[nFrmWds + tid - nDscWds]; } //printf("### Reader[%u]: tid %d, i %lu, hdr %08x\n", reader, tid, nFrmWds + tid - nDscWds, hdr[tid]); }
    //else if (tid < nLdrWds+2) { printf("### Reader[%u]: tid %d, i %lu, %04x %04x\n", reader, tid, nFrmWds + tid - nDscWds, in[nFrmWds + tid - nDscWds] & 0xffff, (in[nFrmWds + tid - nDscWds]>>16)&0xffff); }

    // Calibrate
    //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader[%u]: blk %3d, in[1] %u, th sz %lu\n",
    //                                                  reader, blockIdx.x, in[1], sizeof(TimingHeader));
    auto const payloadSz = in[1] - sizeof(TimingHeader);
    //if (tid == 0)  printf("### Reader[%u]: dmaSz %u, pyldSz %ld\n", reader, in[1], payloadSz);
    if (payloadSz > 0) { // Calibrate only when there's a payload
      auto const __restrict__ raw = (uint16_t*)&in[nFrmWds + nHdrWds];
      //if (tid == 0)  printf("### Reader[%u]: raw %p\n", reader, raw);
      auto const __restrict__ out = &calibBuffers[pblBufIdx * calibBufsCnt];
      //if (tid == 0)  printf("### Reader[%u]: out %p\n", reader, out);
      auto const payloadCnt = payloadSz/sizeof(*raw);
      //if (tid == 0)  printf("### Reader[%u]: payloadCnt %ld, calibBufsCnt %lu\n", reader, payloadCnt, calibBufsCnt);
      auto const elementCnt = payloadCnt > calibBufsCnt ? calibBufsCnt : payloadCnt;
      //if (tid == 0)  printf("### Reader[%u]: payloadCnt %ld, calibBufsCnt %lu, cnt %lu\n",
      //                      reader, payloadCnt, calibBufsCnt, elementCnt);
      auto const __restrict__ ref = refBuffers ? &refBuffers[(pblBufIdx % refBufCnt) * calibBufsCnt] : (float*)0;
      //if (tid == 0)  printf("### Reader[%u]: idx %u, refBuffers %p, ref %p\n", reader, pblBufIdx%refBufCnt, refBuffers, ref);

      //auto const stride  = blockDim.x * gridDim.x;
      //if (tid == 0)  printf("### calibrate: nElements %lu, stride %u, idx %u, calib %p:%p\n",
      //                       elementCnt, stride, pblBufIdx, &out[0],&out[elementCnt]);

      //if (tid == 0) *stateMon = 11;
      _calibrate(out, raw, elementCnt, rangeOffset, rangeBits, pedArray, gainArray, ref);
      //if (calibFn) (*calibFn)(out, raw, elementCnt); // Not working
      //if (tid == 0) *stateMon = 12;
    }

    // Advance to the next state
    // State variable is likely set before the last thread is done, but the
    // next kernel won't check it before all threads of this kernel complete
    if (tid == 0) *state = 3;
  }
}

// This will re-launch the current graph
static __global__
void _readerLoop(unsigned  const                            reader,
                 unsigned  const                            nReaders,
                 unsigned  const                            nRdrShft,      // log2(nReaders)
                 unsigned* const               __restrict__ state,
                 unsigned* const               __restrict__ dmaBufferIdx,
                 unsigned* const               __restrict__ pebbleIdx,
                 uint8_t*  const               __restrict__ wrEnReg,
                 uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
                 size_t    const                            dmaCount,
                 RingIndexDtoD*   const        __restrict__ readerQueue,
                 cuda::std::atomic<unsigned> const&         terminate,
                 uint64_t* const               __restrict__ stateMon,
                 uint64_t* const               __restrict__ evtCtr)
{
  if (*state == 3) {
    DBG(*stateMon = 13;)
    bool rc;
    unsigned ns{8};
    auto const pblIdx{(*pebbleIdx) >> nRdrShft};
    while ( (rc = !readerQueue->push(pblIdx)) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        DBG(*stateMon = 14;)
        break;
      }
    }
    if (!rc) {
      //printf("### Reader[%u]: pushed pblIdx %u, hd %u, tl %u, occ %u\n", reader, *pebbleIdx,
      //       readerQueue->head(), readerQueue->tail(), readerQueue->occupancy());

      auto const dmaIdx{*dmaBufferIdx};
      *(volatile uint32_t*)(dmaBuffers[dmaIdx] + 4) = 0;   // Clear the handshake space
      *(volatile uint8_t*)(wrEnReg + dmaIdx * 4) = 1;      // Enable the DMA
      *dmaBufferIdx = (dmaIdx + nReaders) & (dmaCount-1);  // Prepare for the next DMA buffer
      //printf("### Reader[%u]: idx %u, hand shake %p, next write enable %p\n",
      //       reader, dmaIdx, dmaBuffers[dmaIdx] + 4, wrEnReg + dmaIdx * 4);

      *state = 0;
      DBG(*stateMon = 15; ++(*evtCtr);)
    }
  }

  // Relaunch the graph
  //printf("### Reader[%u]: 2 relaunch\n", reader);
  if (!terminate.load(cuda::std::memory_order_acquire)) {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
  else {
    //printf("### Reader[%u]: Terminate is True: not relaunching\n", reader);
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
cudaGraph_t Reader::_recordGraph(unsigned reader)
{
  rdr_scoped_range r{/*"Reader::_recordGraph"*/}; // Expose function name via NVTX

  auto       stream         = m_streams[reader];
  auto       panel          = m_pool.panel();
  auto const wrEnReg_d      = (uint8_t*)panel->fpgaRegs + panel->coreRegs.freeListOffset(0);
  auto const dmaBuffers_d   = panel->dmaBuffers_d;
  auto const dmaCount       = m_pool.dmaCount();
  auto const frameSize      = panel->coreRegs.dmaDataBytes();
  auto const hostWrtBufs    = m_pool.hostWrtBufs();
  auto const hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs);
  auto const calibBuffers_d = m_pool.calibBuffers_d();
  auto const calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers_d);
  auto const rangeOffset    = m_det.rangeOffset();
  auto const rangeBits      = m_det.rangeBits();
  auto const refBuffers_d   = m_det.referenceBuffers();
  auto const refBufCnt      = m_det.referenceBufCnt();
  auto const pedArray_d     = m_det.pedestals_d();
  auto const gainArray_d    = m_det.gains_d();
  //auto const calibFn_d      = m_det.getCalibFn(); // Not working
  auto const nRdrShft = ffs(m_nReaders) - 1; // log2(nReaders)

  // Determine how many processing resources to reserve for the Reader kernel
  // @todo: The maybe should be done in PgpDetector in conjunction with the other components
  cudaDeviceProp prop;
  chkError(cudaGetDeviceProperties(&prop, 0));
  const auto tpSM{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;
  switch (tpSM) {
    case 1536:  nSMs = 6;  break;
    case 2048:  nSMs = 8;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpSM);
      abort();
  };
  // Slightly better times seem to be achieved when nPixels/stride is an integer
  // Adjusting nBlocks for this might lead to a partially used SM, but aim for
  // maximum occupancy of the SMs
  const auto bpSM{prop.maxBlocksPerMultiProcessor};
  auto nThreads{tpSM/bpSM};
  auto nBlocks{nSMs*bpSM};
  auto stride{nBlocks * nThreads};
  logging::info("GPU threads per SM: %d, blocks per SM: %d, blocks %u * threads %u = %u total threads, SMs %.1f, elements per thread: %.1f\n",
                tpSM, bpSM, nBlocks, nThreads, stride, float(stride) / tpSM, float(calibBufsCnt) / stride);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream begin-capture failed")) {
    return 0;
  }

  _waitForDMA<<<1, 1, 1, stream>>>(reader,
                                   m_nReaders,
                                   m_states_d[reader],
                                   m_dmaBufferIdxes[reader],
                                   m_pebbleIdxes[reader],
                                   dmaBuffers_d,
                                   m_pebbleQueue.d,
                                   m_metrics.states[reader],
                                   m_metrics.pblWtCtrs[reader],
                                   m_metrics.dmaWtCtrs[reader]);
  chkError(cudaGetLastError(), "Launch of _waitForDMA kernel failed");

  // Copy the DMA descriptor and the timing header to host-visible pinned memory buffers
  // Calibrate the raw data from the DMA buffers into the calibrated data buffers
  //constexpr auto iPayload { (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(uint32_t) };
  _event<<<nBlocks, nThreads, 0, stream>>>(reader,
                                           m_states_d[reader],
                                           m_dmaBufferIdxes[reader],
                                           m_pebbleIdxes[reader],
                                           dmaBuffers_d,
                                           frameSize,
                                           hostWrtBufs,
                                           hostWrtBufsCnt,
                                           calibBuffers_d,
                                           calibBufsCnt,
                                           pedArray_d,
                                           gainArray_d,
                                           //calibFn_d, // Not working
                                           rangeOffset,
                                           rangeBits,
                                           refBuffers_d,
                                           refBufCnt,
                                           m_metrics.states[reader]);
  chkError(cudaGetLastError(), "Launch of _event kernel failed");

  // Publish the current head index and re-launch
  _readerLoop<<<1, 1, 0, stream>>>(reader,
                                   m_nReaders,
                                   nRdrShft,
                                   m_states_d[reader],
                                   m_dmaBufferIdxes[reader],
                                   m_pebbleIdxes[reader],
                                   wrEnReg_d,
                                   dmaBuffers_d,
                                   dmaCount,
                                   m_readerQueues[reader].d,
                                   m_terminate_d,
                                   m_metrics.states[reader],
                                   m_metrics.fwdWtCtrs[reader]);
  chkError(cudaGetLastError(), "Launch of _readerLoop kernel failed");

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph), "Stream end-capture failed")) {
    return 0;
  }

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

  // Enable a DMA for all buffers
  for (unsigned dmaBufIdx = 0; dmaBufIdx < m_pool.dmaCount(); ++dmaBufIdx) {
    // Clear handshake space on the GPU side (A.K.A. "GPU's doorbell")
    const auto dmaBufs = panel->dmaBuffers[dmaBufIdx];
    chkError(cudaMemset(dmaBufs + 4, 0, sizeof(uint32_t)));
    //const auto dmaWrtStart = (uint8_t*)panel->fpgaRegs + panel->coreRegs.freeListOffset(dmaBufIdx);
    //printf("*** Reader: dmaBufIdx %u, dmaBuffer %p, hwWrtStart %p\n", dmaBufIdx, dmaBufs, dmaWrtStart);

#ifndef HOST_REARMS_DMA
    // Write to the DMA start register in the FPGA to trigger the write
    panel->coreRegs.returnFreeListIndex(dmaBufIdx);
#endif // HOST_REARMS_DMA
  }

  // Ensure that the DMA round-robin index starts with buffer 0
  dmaIdxReset(panel->coreRegs);

  // Launch the Reader graph
  for (unsigned i = 0; i < m_nReaders; ++i) {
    //printf("*** Reader: Launching graph %u\n", i);
    chkFatal(cudaGraphLaunch(m_graphExecs[i], m_streams[i]));
  }
}

void Reader::freeDma(PGPEvent* event)
{
  //printf("*** Reader::freeDma: evt %p, idx %u, pgpIdx %zu\n", event, event->buffers[0].index, event - &m_pool.pgpEvents[0]);
  constexpr uint32_t lane{0};                // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  event->mask = 0;
  m_pebbleQueue.h->push(buffer->index);
  //m_pool.freeDma(1, nullptr); // This doesn't do anything
  //printf("*** Reader::freeDma: pblIdx %u, hd %u, tl %u, occ %u\n", buffer->index, m_pebbleQueue.h->head(), m_pebbleQueue.h->tail(), m_pebbleQueue.h->occupancy());
}

void Reader::flush()
{
  // Free buffers associated with the DMAs
  for (auto& event : m_pool.pgpEvents) {
    if (event.mask)  freeDma(&event);   // Leaves event mask = 0
  }

  // Free any in-use pebble buffers
  m_pool.flushPebble();
}

