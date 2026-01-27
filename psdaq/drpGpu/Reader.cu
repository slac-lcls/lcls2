#include "Reader.hh"

#include "Detector.hh"
#include "drp/spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"

#include <thread>

//#include <cuda/barrier>
#include <cooperative_groups.h>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;

struct rdr_domain{ static constexpr char const* name{"Reader"}; };
using rdr_scoped_range = nvtx3::scoped_range_in<rdr_domain>;

//using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cg = cooperative_groups;

Reader::Reader(const Parameters&                  para,
               MemPoolGpu&                        pool,
               Detector&                          det,
               size_t                             trgPrimitiveSize,
               const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool       (pool),
  m_terminate_d(terminate_d),
  m_para       (para)
{
  m_det.h = &det;
  chkError(cudaMalloc(&m_det.d,          sizeof(*m_det.d)));
  chkError(cudaMemcpy( m_det.d, m_det.h, sizeof(*m_det.d), cudaMemcpyHostToDevice));

  // Set up buffer index allocator for DMA to Collector comms
  m_readerQueue.h = new RingIndexDtoD(m_pool.nbuffers(), m_terminate_d);
  chkError(cudaMalloc(&m_readerQueue.d,                  sizeof(*m_readerQueue.d)));
  chkError(cudaMemcpy( m_readerQueue.d, m_readerQueue.h, sizeof(*m_readerQueue.d), cudaMemcpyHostToDevice));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo};
  logging::debug("Reader stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Allocate a stream at lowest priority so that higher priority can
  // be given to downstream stages that help drain the system
  chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));

  // Keep track of the head index of the Reader stream
  chkError(cudaMalloc(&m_head,    sizeof(*m_head)));
  chkError(cudaMemset( m_head, 0, sizeof(*m_head)));

  // Gather up DMA buffer pointers into a device-usable array
  auto nFpgas{m_pool.panels().size()};
  auto dmaCount{m_pool.dmaCount()};
  chkError(cudaMalloc(&m_dmaBuffers, nFpgas * dmaCount * sizeof(*m_dmaBuffers)));
  chkError(cudaMalloc(&m_swFpgaRegs, nFpgas * sizeof(*m_swFpgaRegs)));
  for (unsigned i = 0; i < nFpgas; ++i) {
    const auto& fpga = m_pool.panels()[i];
    for (unsigned j = 0; j < dmaCount; ++j) {
      printf("*** Reader: fpga %u, dmaBufIdx %u, hwWrtPtr %p, hwWrtStart %p\n",
             i, j, (void*)(fpga.dmaBuffers[j].dptr), (void*)(fpga.swFpgaRegs.dptr + GPU_ASYNC_WR_ENABLE(j)));
      chkError(cudaMemcpy(&m_dmaBuffers[i * dmaCount + j], &fpga.dmaBuffers[j].dptr, sizeof(*m_dmaBuffers), cudaMemcpyHostToDevice));
    }
    chkError(cudaMemcpy(&m_swFpgaRegs[i], &fpga.swFpgaRegs.dptr, sizeof(*m_swFpgaRegs), cudaMemcpyHostToDevice));
  }

  // Prepare buffers visible to the host for receiving headers
  const size_t bufSz = sizeof(DmaDsc)+sizeof(TimingHeader) + trgPrimitiveSize;
  for (unsigned fpga = 0; fpga < nFpgas; ++fpga) {
    m_pool.createHostBuffers(fpga, bufSz);
  }

  // Prepare the CUDA graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Reader graph");
    abort();
  }
}

Reader::~Reader()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  auto nFpgas{m_pool.panels().size()};
  for (unsigned fpga = 0; fpga < nFpgas; ++fpga) {
    m_pool.destroyHostBuffers(fpga);
  }

  if (m_head)  chkError(cudaFree(m_head));

  chkError(cudaStreamDestroy(m_stream));

  if (m_readerQueue.d)  chkError(cudaFree(m_readerQueue.d));
  if (m_readerQueue.h)  delete m_readerQueue.h;

  if (m_det.d)  chkError(cudaFree(m_det.d));
}

int Reader::_setupGraph()
{
  cudaGraphExec_t& graphExec = m_graphExec;
  cudaStream_t     stream    = m_stream;

  // Generate the graph
  logging::debug("Recording Reader graph");
  auto graph = _recordGraph();
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

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reader graph...");
  if (chkError(cudaGraphUpload(graphExec, stream), "Reader graph upload failed")) {
    return -1;
  }

  return 0;
}

static __device__
void _calibrate(Gpu::Detector const& detector,
                float*        const  calib,
                uint16_t*     const  raw,
                unsigned      const  count,
                unsigned      const  nPanels,
                unsigned      const  rangeOffset,
                unsigned      const  rangeBits)
{
  auto const tid     = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride  = gridDim.x * blockDim.x;
  //if (tid == 0)  printf("*** Reader: tid %d, stride %d, nPanels %u\n", tid, stride, nPanels);
  auto const panel   = tid / (stride / nPanels);  // Split the panel handling evenly across the allocated threads
  //if (tid == 0)  printf("*** Reader: panel %u\n", panel);
  auto const nPixels = count / nPanels;

  //if (tid == 0)  printf("*** Reader: m_pedArr_d %p\n", detector.m_pedArr_d);
  auto const pedArr  = detector.m_pedArr_d[panel];
  //if (tid == 0)  printf("*** Reader: pedArr %p\n", pedArr);
  //if (tid == 0)  printf("*** Reader: m_gainArr_d %p\n", detector.m_gainArr_d);
  auto const gainArr = detector.m_gainArr_d[panel];
  //if (tid == 0)  printf("*** Reader: gainArr %p\n", gainArr);
  //if (tid == 0)  printf("*** Reader: count %u, stride %d, loops %d\n", count, stride, count / stride);
  auto const rangeMask{1 << rangeBits - 1};
  auto const dataMask {1 << rangeOffset - 1};
  for (auto i = tid; i < count; i += stride) {
    auto const range = (raw[i] >> rangeOffset) & rangeMask;
    auto const peds  = &pedArr [range * nPixels];
    auto const gains = &gainArr[range * nPixels];
    auto const data  = raw[i] & dataMask;
    calib[panel * nPixels + i] = (data - peds[i]) * gains[i];
  }
  //if (tid == 0)  printf("*** Reader: calibrate returning\n");
}

static __global__
void _handleDMA(CUdeviceptr* const        __restrict__ swFpgaRegs,    // [nFpgas]
                size_t       const                     nFpgas,
                CUdeviceptr* const        __restrict__ dmaBuffers,    // [nFpgas * dmaCount][maxDmaSize]
                size_t       const                     dmaCount,
                uint32_t*    const* const __restrict__ outBufs,
                size_t       const                     outBufsCnt,
                float*       const        __restrict__ calibBuffers,
                size_t       const                     calibBufsCnt,
                unsigned&                              dmaBufferIdx,
                RingIndexDtoD&                         readerQueue,
                Gpu::Detector const&                   detector,
                unsigned      const                    rangeOffset,
                unsigned      const                    rangeBits,
                cuda::std::atomic<unsigned> const&     terminate)
{
  cg::thread_block cta = cg::this_thread_block();

  __shared__ bool     done;
  __shared__ unsigned dmaBufIdx;
  __shared__ unsigned pblBufIdx;

  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = gridDim.x * blockDim.x;

  // Get a pointer to DMA buffers for each datadev
  auto const fpga = tid / (stride / nFpgas);  // Split the FPGA handling evenly across the allocated threads
  //if (tid == 0)  printf("### Reader: blkDim %d, gridDim %d, fpga %lu, dmaCnt %lu\n",
  //                      blockDim.x, gridDim.x, fpga, dmaCount);
  auto const __restrict__ dmaBufs = &dmaBuffers[fpga * dmaCount];
  //if (tid == 0)  printf("### Reader: dmaBufs %p, done %u, calibBufsCnt %lu\n", dmaBufs, done, calibBufsCnt);

#ifdef PERSISTENT_KERNEL
  // Enable the first DMA
  if (threadIdx.x == 0) {
    if (blockIdx.x < nFpgas) {
      //printf("### Reader: dmaBufs[%u] %p, sz %p, hwWrtStart[%lu] %p\n",
      //       dmaBufferIdx, (void*)(dmaBufs[dmaBufferIdx]), (void*)(dmaBufs[dmaBufferIdx]+4), fpga, (void*)(swFpgaRegs[fpga]+GPU_ASYNC_WR_ENABLE(dmaBufferIdx)));
      *(uint32_t*)(dmaBufs[dmaBufferIdx] + 4) = 0; // Clear the handshake space of the first DMA buffer
      *(uint8_t*)(swFpgaRegs[fpga] + GPU_ASYNC_WR_ENABLE(dmaBufferIdx)) = 1; // Enable the DMA on this dataDev
    }
  }

  do
#endif // PERSISTENT_KERNEL
  {
    if (threadIdx.x == 0) {
      done = terminate.load(cuda::std::memory_order_acquire);
      //if (tid == 0)  printf("### Reader: done %u\n", done);
      if (done)  return; //continue;

      dmaBufIdx = dmaBufferIdx;         // Load shmem buf idx from global memory for all blocks
      //if (tid == 0)  printf("### Reader: dmaBufIdx %u\n", dmaBufIdx);

      if (blockIdx.x < nFpgas) {
        // Allocate the index of the next set of intermediate buffers to be used
        //if (tid == 0)  printf("### Reader: get pblBufIdx\n");
        pblBufIdx = readerQueue.allocate();     // This blocks when no buffers available
        //if (tid == 0)  printf("### Reader: pblBufIdx %u\n", pblBufIdx);

        const volatile uint32_t* __restrict__ mem = (uint32_t*)(dmaBufs[dmaBufIdx] + 4);
        //if (tid == 0)  printf("### Reader: Wait for DMA[%u] from FPGA %lu: mem %p\n", dmaBufIdx, fpga, mem);
        unsigned ns = 8;
        while (*mem == 0) {                     // Wait for DMA completion @todo: abort by writing this location
          done = terminate.load(cuda::std::memory_order_acquire);
          if (done)  return;
          __nanosleep(ns);
          if (ns < 256)  ns *= 2;
        }
        //if (tid == 0)  printf("### Reader: Got DMA[%u]: sz %u, TH[0] %016lx\n", dmaBufIdx, *mem, *((uint64_t*)(&mem[8-1])));
        auto next = (dmaBufIdx + 1) % dmaCount; // Prepare for next DMA buffer
        *(uint32_t*)(dmaBufs[next] + 4) = 0;    // Clear the handshake space of the next DMA buffer
        *(uint8_t*)(swFpgaRegs[fpga] + GPU_ASYNC_WR_ENABLE(next)) = 1;  // Enable the DMA on this dataDev
        if (blockIdx.x == 0)                    // Update global memory  @todo: check for race
          dmaBufferIdx = next;                  // only once (same value for all blocks)
        //if (tid == 0)  printf("### Reader: next %lu\n", next);
      }

      // @todo: Event build: Check DMA sizes and PIDs are all the same
    }
    cg::sync(cta); // Block all threads until DMA completes and shmem is updated

    // Save the DMA descriptor and TimingHeader in pinned memory
    //if (tid == 0)  printf("### Reader: dmaIdx %u, dmaBufs %p\n", dmaBufIdx, dmaBufs);
    auto const __restrict__ in  = (uint32_t*)dmaBufs[dmaBufIdx];
    //if (tid == 0)  printf("### Reader: dmaIdx %u, in %p\n", dmaBufIdx, in);
    //if (tid == 0)  printf("### Reader: fpga %lu, outBufs %p\n", fpga, outBufs);
    auto const __restrict__ hdr = outBufs[fpga] + pblBufIdx * outBufsCnt;
    //if (tid == 0)  printf("### Reader: ob %p, pblIdx %u, outBufsCnt %lu, hdr %p\n", outBufs[fpga], pblBufIdx, outBufsCnt, hdr);
    constexpr auto nHdrWords = (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(*in);
    auto const i    = tid % nHdrWords;
    auto const tid0 = fpga * (stride / nFpgas);
    //if (tid == 0)  printf("### Reader: nHdrWords %lu, i %lu, tid0 %lu\n", nHdrWords, i, tid0);
    if (tid >= tid0 && tid < tid0 + nHdrWords) {
      hdr[i] = in[i];
      //printf("### Reader: hdr[%lu] %08x\n", i, in[i]);
    }

    // Calibrate
    //if (tid == 0)  printf("### Reader: in[1] %u, th sz %lu\n", in[1], sizeof(TimingHeader));
    if (in[1] > sizeof(TimingHeader)) {
      auto const __restrict__ raw = (uint16_t*)&in[nHdrWords];
      //if (tid == 0)  printf("### Reader: raw %p\n", raw);
      auto const __restrict__ out = calibBuffers + pblBufIdx * calibBufsCnt;
      //if (tid == 0)  printf("### Reader: out %p\n", out);
      auto const payloadCnt = (in[1] - sizeof(TimingHeader))/sizeof(*raw);
      //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu\n", payloadCnt, calibBufsCnt);
      auto const count = payloadCnt > calibBufsCnt ? calibBufsCnt : payloadCnt; // @todo: Alert to truncation
      //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu, cnt %lu, nFpgas %lu\n",
      //                      payloadCnt, calibBufsCnt, count, nFpgas);

      _calibrate(detector, out, raw, count, nFpgas, rangeOffset, rangeBits);
      //if (tid == 0)  printf("### Reader: calib done\n");
    }

    //if (tid == 0)  printf("### Reader: posting %u\n", pblBufIdx);
    if (tid == 0)  readerQueue.post(pblBufIdx);
    //if (tid == 0)  printf("### Reader: posted  %u\n", pblBufIdx);
  }
#ifdef PERSISTENT_KERNEL
  while (!done);
  //if (tid == 0)  printf("### Reader: returning\n");
#else // Relaunched graph

  // Relaunch the graph
  if (tid == 0)  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
#endif // PERSISTENT_KERNEL
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

  auto stream               = m_stream;
  auto nFpgas               = m_pool.panels().size();
  auto const hostWrtBufs_d  = m_pool.hostWrtBufs_d();
  auto const hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(**hostWrtBufs_d);
  auto const calibBuffers   = m_pool.calibBuffers_d();
  auto const calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers);

  // Determine how many processing resources to reserve for the Reader kernel
  // @todo: The maybe should be done in PgpDetector in conjunction with the other components
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const auto tpMP{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;                  // @todo: Maybe allow nSMs to be overridable?
  switch (tpMP) {
    case 1536:  nSMs = 4;  break;
    case 2048:  nSMs = 2;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpMP);
      abort();
  };
  // Slightly better times seem to be achieved when nPixels/stride is an integer
  // Adjusting nBlocks for this might lead to a partially used SM, but aim for
  // maximum occupancy of the SMs
  unsigned nThreads{32}; // @todo: Should come from para.nGpuThreads or a kwarg?
  unsigned nBlocks{189}; //{nSMs * tpMP / nThreads}; // {189};
  unsigned stride{nBlocks * nThreads};

  logging::info("GPU threads per SM: %d, total threads: %u, SMs %.1f, elements per thread: %.1f\n",
                tpMP, stride, float(stride) / tpMP, float(calibBufsCnt) / stride);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream begin-capture failed")) {
    return 0;
  }

  _handleDMA<<<nBlocks, nThreads, 0, m_stream>>>(m_swFpgaRegs,
                                                 nFpgas,
                                                 m_dmaBuffers,
                                                 m_pool.dmaCount(),
                                                 hostWrtBufs_d,
                                                 hostWrtBufsCnt,
                                                 calibBuffers,
                                                 calibBufsCnt,
                                                 *m_head, // @todo: Fix name?
                                                 *m_readerQueue.d,
                                                 *m_det.d,
                                                 m_det.h->rangeOffset(),
                                                 m_det.h->rangeBits(),
                                                 m_terminate_d);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph), "Stream end-capture failed")) {
    return 0;
  }

  printf("*** Reader: Returning graph\n");
  return graph;
}

void Reader::start()
{
  logging::info("Reader is starting");
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  for (const auto& fpga : m_pool.panels()) {
    if (fpga.name != "/dev/null") {     // Else, Simulator mode
      // Ensure that timing messages are DMAed to the GPU
      dmaTgtSet(fpga.datadev, DmaTgt_t::TGT_GPU);

      // Ensure that the DMA round-robin index starts with buffer 0
      dmaIdxReset(fpga.datadev);
    }

#ifdef HOST_REARMS_DMA
    // Write to the DMA start register in the FPGA
    for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
      auto rc = gpuSetWriteEn(fpga.datadev.fd(), dmaIdx);
      if (rc < 0) {
        logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
        abort();
      }
    }
#endif // HOST_REARMS_DMA
  }

#ifndef PERSISTENT_KERNEL
  // Enable a DMA for buffer 0 only
  unsigned instance{0};
  for (auto& panel: m_pool.panels()) {
    /****************************************************************************
     * Clear the handshake space
     * Originally was cuStreamWriteValue32, but the stream functions are not
     * supported within graphs. cuMemsetD32Async acts as a good replacement.
     ****************************************************************************/
    chkError(cuMemsetD32Async(panel.dmaBuffers[instance].dptr + 4, 0, 1, m_stream));
    printf("*** instance %u, dmaBuffer %p\n", instance, (void*)panel.dmaBuffers[instance].dptr);

#ifndef HOST_REARMS_DMA
    // Write to the DMA start register in the FPGA to trigger the write
    chkError(cuMemsetD8Async(panel.swFpgaRegs.dptr + GPU_ASYNC_WR_ENABLE(instance), 1, 1, m_stream));
    printf("*** instance %u, hwWriteStart %p\n", instance, (void*)(panel.swFpgaRegs.dptr + GPU_ASYNC_WR_ENABLE(instance)));
#endif // HOST_REARMS_DMA
  }
#endif // PERSISTENT_KERNEL

  // Launch the Reader graph
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}
