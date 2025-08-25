#include "Reader.hh"

#include "Detector.hh"
#include "drp/spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"

#include <thread>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


Reader::Reader(unsigned                 panel,
               const Parameters&        para,
               MemPoolGpu&              pool,
               Detector&                det,
               size_t                   trgPrimitiveSize,
               const cuda::atomic<int>& terminate_d) :
  m_pool         (pool),
  m_det          (det),
  m_terminate_d  (terminate_d),
  m_panel        (panel),
  m_para         (para)
{
  //printf("*** Reader ctor 1\n");

  // Set up buffer index allocator for DMA to Collector comms
  m_readerQueue.h = new Gpu::RingIndexDtoD(m_pool.nbuffers(), m_pool.dmaCount(), m_terminate_d);
  chkError(cudaMalloc(&m_readerQueue.d,                  sizeof(*m_readerQueue.d)));
  chkError(cudaMemcpy( m_readerQueue.d, m_readerQueue.h, sizeof(*m_readerQueue.d), cudaMemcpyHostToDevice));

  // Allocate a stream per buffer
  m_streams.resize(m_pool.dmaCount());
  for (auto& stream : m_streams) {
    chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  //printf("*** Reader ctor 2, %zu streams\n", m_streams.size());

  // Set up a done flag to cache m_terminate's value and avoid some PCIe transactions
  chkError(cudaMalloc(&m_done,    sizeof(*m_done)));
  chkError(cudaMemset( m_done, 0, sizeof(*m_done)));
  //printf("*** Reader ctor 3\n");

  // Keep track of the head index of each Reader stream
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    chkError(cudaMalloc(&m_head[i],    sizeof(*m_head[i])));
    chkError(cudaMemset( m_head[i], 0, sizeof(*m_head[i])));
  }
  //printf("*** Reader ctor 4: trgPrmtv sz %zu\n", trgPrimitiveSize);

  // Prepare buffers visible to the host for receiving headers
  const size_t bufSz = sizeof(DmaDsc)+sizeof(TimingHeader) + trgPrimitiveSize;
  //printf("*** Reader ctor 4a: sz %zu, nbufs %u\n", bufSz, m_pool.nbuffers());
  m_pool.createHostBuffers(panel, bufSz);
  //printf("*** Reader ctor 5\n");

  // Prepare the CUDA graphs
  m_graphExecs.resize(m_pool.dmaCount());
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    if (_setupGraphs(i)) {
      logging::critical("Failed to set up Reader[%u] graphs", panel);
      abort();
    }
  }
  //printf("*** Reader ctor 6\n");
}

Reader::~Reader()
{
  printf("*** Reader dtor 1\n");
  for (auto& graphExec : m_graphExecs) {
    chkError(cudaGraphExecDestroy(graphExec));
  }
  printf("*** Reader dtor 2\n");

  m_pool.destroyHostBuffers(m_panel);
  printf("*** Reader dtor 3\n");

  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    chkError(cudaFree(m_head[i]));
  }
  chkError(cudaFree(m_done));
  printf("*** Reader dtor 4\n");

  for (auto& stream : m_streams) {
    chkError(cudaStreamDestroy(stream));
  }
  printf("*** Reader dtor 5\n");

  if (m_readerQueue.d)  chkError(cudaFree(m_readerQueue.d));
  delete m_readerQueue.h;
  printf("*** Reader dtor 6\n");
}

int Reader::_setupGraphs(unsigned instance)
{
  cudaGraph_t      graph;
  cudaGraphExec_t& graphExec = m_graphExecs[instance];
  cudaStream_t     stream    = m_streams[instance];

  // Generate the graph
  //printf("*** Reader setupGraphs 1.%u\n", instance);
  auto& hostWriteBufs = m_pool.hostWrtBufsVec_h()[m_panel][0]; // [0] is the base ptr of the whole allocation
  //printf("*** Reader setupGraphs 2.%u\n", instance);
  // Attach host-visible memory to the stream
  chkError(cudaStreamAttachMemAsync(stream, hostWriteBufs, 0, cudaMemAttachHost));
  //printf("*** Reader setupGraphs 3.%u\n", instance);
  logging::debug("Recording Reader graph %d", instance);
  const auto& panel = m_pool.panels()[m_panel];
  graph = _recordGraph(instance, panel.dmaBuffers[instance].dptr, panel.hwWriteStart);
  //printf("*** Reader setupGraphs 4.%u\n", instance);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the graph. The resulting CUgraphExec may only be executed once
  // at any given time.  I believe it can be reused, but it cannot be launched
  // while it is already running.  If we wanted to launch multiple, we would
  // instantiate multiple CUgraphExec's and then launch those individually.
  if (chkError(cudaGraphInstantiate(&graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reader graph create failed")) {
    return -1;
  }
  //printf("*** Reader setupGraphs 5.%u\n", instance);

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reader graph %u...", instance);
  if (chkError(cudaGraphUpload(graphExec, stream), "Reader graph upload failed")) {
    return -1;
  }
  //printf("*** Reader setupGraphs 6.%u\n", instance);

  return 0;
}

// Wait for the DMA size word to become non-zero
static __global__ void _waitForDMA(const volatile uint32_t* __restrict__ mem,
                                   unsigned                              instance,
                                   Gpu::RingIndexDtoD&                   readerQueue,
                                   unsigned*                __restrict__ head,
                                   const cuda::atomic<int>&              terminate,
                                   bool*                    __restrict__ done)
{
  //printf("### Reader waitForDMA 1.%u\n", instance);

  // Allocate the index of the next set of intermediate buffers to be used
  *head = readerQueue.prepare(instance);
  //printf("### Reader waitForDMA 1.%u, idx %u\n", instance, *head);

  // Wait for data to be DMAed
  //printf("### Reader waitForDMA 2.%u, mem %p\n", instance, mem);
  while (*mem == 0) {
    if ( (*done = terminate.load(cuda::memory_order_acquire)) )  break;
  }
  //printf("### Reader waitForDMA 3.%u, *mem %08x, done = %u\n", instance, *mem, *done);
}

// This copies the DmaDsc and TimingHeader into a host-visible buffer
static __global__ void _event(uint32_t** const __restrict__ outBufs,
                              uint32_t*  const __restrict__ in,
                              unsigned                      instance,
                              const unsigned&               idx,
                              const bool&                   done)
{
  //printf("### Reader _event 1.%u, done %d, idx %u\n", instance, done, idx);
  if (done)  return;
  //printf("### Reader _event 1.%ua, done %d, idx %u, outBufs %p\n", instance, done, idx, outBufs);
  //printf("### Reader _event 1.%ub, done %d, idx %u, *outBufs %p\n", instance, done, idx, *outBufs);

  uint32_t* const __restrict__ out = outBufs[idx];
  //if (threadIdx.x == 0)  printf("### Reader _event 1.%uc, done %d, idx %u, pid %08x%08x\n", instance, done, idx, in[9], in[8]);

  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto nWords = (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(*out);
  //printf("### Reader _event 2.%u, offset %d, nWords %lu\n", instance, offset, nWords);
  for (int i = offset; i < nWords; i += blockDim.x * gridDim.x) {
    out[i] = in[i];
  }
  //printf("### Reader _event 3.%u\n", instance);
}

// This will re-launch the current graph
static __global__ void _graphLoop(const unsigned&     idx,
                                  Gpu::RingIndexDtoD& readerQueue,
                                  const bool&         done)
{
  //printf("### Reader graphLoop 1, done %d, idx %u\n", done, idx);
  if (done)  return;
  //printf("### Reader graphLoop 1a, idx %u\n", idx);

  readerQueue.produce(idx);
  //printf("### Reader graphLoop 2, idx %u\n", idx);

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  //printf("### Reader graphLoop 3\n");
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
cudaGraph_t Reader::_recordGraph(unsigned    instance,
                                 CUdeviceptr dmaBuffer,
                                 CUdeviceptr hwWriteStart)
{
  //printf("*** Reader record 1.%d\n", instance);
  auto stream          = m_streams[instance];
  auto hostWriteBufs_d = m_pool.hostWrtBufsVec_d()[m_panel];
  auto hostWriteBufsSz = m_pool.hostWrtBufsSize();
  //printf("*** Reader record 2.%d, hostWriteBufs_d[%u] %p, sz %zu\n",
  //       instance, m_panel, hostWriteBufs_d, hostWriteBufsSz);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream begin-capture failed")) {
    return 0;
  }

  /****************************************************************************
   * Clear the handshake space
   * Originally was cuStreamWriteValue32, but the stream functions are not
   * supported within graphs. cuMemsetD32Async acts as a good replacement.
   ****************************************************************************/
  chkError(cuMemsetD32Async(dmaBuffer + 4, 0, 1, stream));

  // Wipe the buffer (for debugging; normally commented out for performance)
  //chkError(cuMemsetD32Async(dmaBuffer, 0, m_pool.dmaSize() / 4, stream));

#ifndef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA to trigger the write
  chkError(cuMemsetD8Async(hwWriteStart + 4 * instance, 1, 1, stream));
#endif

  /*****************************************************************************
   * Spin on the handshake location until the value is non-zero
   * This waits for the data to arrive before starting the processing
   * Originally this was a call to cuStreamWait, but that is not supported by
   * graphs, so instead we use a waitForDMA kernel to spin on the location
   * until data is ready to be processed.
   * @todo: This may have negative implications on GPU scheduling.  Profile it!
   ****************************************************************************/
  _waitForDMA<<<1, 1, 1, stream>>>((uint32_t*)(dmaBuffer + 4),
                                   instance,
                                   *m_readerQueue.d,
                                   m_head[instance],
                                   m_terminate_d,
                                   m_done);
  //printf("*** Reader record 3.%d\n", instance);

  // Copy the DMA descriptor and the timing header to host-visible managed memory buffers
  constexpr auto iPayload { (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(uint32_t) };
  _event<<<1, iPayload, 0, stream>>>(hostWriteBufs_d, (uint32_t*)dmaBuffer, instance, *m_head[instance], *m_done);
  //printf("*** Reader record 4.%d\n", instance);

  // Calibrate the raw data from the DMA buffers into the calibrated data buffers
  m_det.recordGraph(stream, *m_head[instance], m_panel, (uint16_t*)(dmaBuffer + iPayload));
  //printf("*** Reader record 5.%d\n", instance);

  // Publish the current head index and re-launch
  _graphLoop<<<1, 1, 0, stream>>>(*m_head[instance], *m_readerQueue.d, *m_done);
  //printf("*** Reader record 6.%d\n", instance);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph), "Stream end-capture failed")) {
    return 0;
  }

  return graph;
}

void Reader::start()
{
  logging::info("Reader[%d] starting", m_panel);
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  const auto& panel = m_pool.panels()[m_panel];

  // Ensure that timing messages are DMAed to the GPU
  dmaTgtSet(panel.gpu, DmaTgt_t::GPU);
  //printf("*** Reader[%d] start 1\n", m_panel);

  // Ensure that the DMA round-robin index starts with buffer 0
  dmaIdxReset(panel.gpu);
  //printf("*** Reader[%d] start 2\n", m_panel);

#ifdef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA
  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    //printf("*** Reader[%d] start: Enable write to DMA buffer %u\n", m_panel, dmaIdx);
    auto rc = gpuSetWriteEn(panel.gpu.fd(), dmaIdx);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
      abort();
    }
  }
#endif
  //printf("*** Reader[%d] start 3\n", m_panel);

  // Launch the DMA graphs
  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    chkFatal(cudaGraphLaunch(m_graphExecs[dmaIdx], m_streams[dmaIdx]));
  }
  //printf("*** Reader[%d] start 4\n", m_panel);
}
