#include "Reader.hh"

#include "Detector.hh"
#include "drp/spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <thread>

#include <cooperative_groups.h>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;

struct rdr_domain{ static constexpr char const* name{"Reader"}; };
using rdr_scoped_range = nvtx3::scoped_range_in<rdr_domain>;

namespace cg = cooperative_groups;

Reader::Reader(const Parameters&                  para,
               MemPoolGpu&                        pool,
               Detector&                          det,
               size_t                             trgPrimitiveSize,
               const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool       (pool),
  m_det        (det),
  m_terminate_d(terminate_d),
  m_para       (para)
{
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

  const auto panel = m_pool.panel();
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    printf("*** Reader: dmaBufIdx %u, hwWrtPtr %p, hwWrtStart %p\n",
           i, (void*)(panel->dmaBuffers[i]), (uint8_t*)panel->fpgaRegs.d + panel->coreRegs.freeListOffset(i));
  }

  // Prepare buffers visible to the host for receiving headers
  const size_t bufSz = sizeof(DmaDsc) + sizeof(TimingHeader) + trgPrimitiveSize;
  m_pool.createHostBuffers(bufSz);

  // Prepare the CUDA graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Reader graph");
    abort();
  }
}

Reader::~Reader()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  m_pool.destroyHostBuffers();

  chkError(cudaStreamDestroy(m_stream));

  if (m_readerQueue.d)  chkError(cudaFree(m_readerQueue.d));
  if (m_readerQueue.h)  delete m_readerQueue.h;
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

  // Instantiate the graph
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
void _calibrate(float*        const        __restrict__ calib,
                uint16_t      const* const __restrict__ raw,
                unsigned      const                     nElements,
                unsigned      const                     rangeOffset,
                unsigned      const                     rangeBits,
                float         const* const __restrict__ pedArray,
                float         const* const __restrict__ gainArray,
                float         const* const __restrict__ ref)
{
  auto const tid     = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride  = blockDim.x * gridDim.x;
  //if (tid == 0)  printf("*** Reader: tid %d, stride %d\n", tid, stride);

  //if (tid == 0)  printf("*** Reader: pedArray %p\n", pedArray);
  //if (tid == 0)  printf("*** Reader: gainArray %p\n", gainArray);
  //if (tid == 0)  printf("*** Reader: count %u, stride %d, loops %d\n", count, stride, count / stride);
  auto const rangeMask{(1 << rangeBits) - 1};
  auto const dataMask {(1 << rangeOffset) - 1};
  for (auto i = tid; i < nElements; i += stride) {
    auto const              range = (raw[i] >> rangeOffset) & rangeMask;
    auto const __restrict__ peds  = &pedArray [range * nElements];
    auto const __restrict__ gains = &gainArray[range * nElements];
    auto const              data  = raw[i] & dataMask;
    calib[i] = (float(data) - peds[i]) * gains[i];

    //if (i < 4) {
    //  printf("### tid %u, i %u: raw %04x, dat %u, rng %u, ped %f, gn %f, cal %f, ref %p: %f\n", tid, i, raw[i], data, range, peds[i], gains[i], calib[i], ref, ref ? ref[i] : 0.f);
    //}
    //if (ref && (calib[i] != ref[i])) {
    //  printf("### blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n", blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
    //}
  }
  //if (tid == 0)  printf("*** Reader: calibrate returning\n");
}

static __device__ bool     indicesValid = false;
static __device__ unsigned dmaBufferIdx = 0;
static __device__ unsigned pebbleIdx    = 0;
static __device__ unsigned blockCount1  = 0;
static __shared__ bool     isLastBlockDone1;
static __device__ unsigned blockCount2  = 0;
static __shared__ bool     isLastBlockDone2;

static __global__
void _handleDMA(uint8_t*  const               __restrict__ wrEnReg,
                uint8_t   const* const* const __restrict__ dmaBuffers,    // [dmaCount][maxDmaSize]
                size_t    const                            dmaCount,
                size_t    const                            frameSize,
                uint32_t* const               __restrict__ hdrBuffers,    // [nBuffers * hdrBufsCnt]
                size_t    const                            hdrBufsCnt,
                float*    const               __restrict__ calibBuffers,  // [nBuffers * calibBufsCnt]
                size_t    const                            calibBufsCnt,
                RingIndexDtoD*   const        __restrict__ readerQueue,
                float     const* const        __restrict__ pedArray,
                float     const* const        __restrict__ gainArray,
                unsigned  const                            rangeOffset,
                unsigned  const                            rangeBits,
                float     const* const        __restrict__ refBuffers,
                unsigned  const                            refBufCnt,
                cuda::std::atomic<unsigned> const&         terminate)
{
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto const __restrict__ dmaBufs = &dmaBuffers[0];

  auto dmaBufIdx = dmaBufferIdx;           // All threads load DMA idx into a register from global memory

  if (threadIdx.x == 0) {                  // Thread 0 of each block
    if (blockIdx.x == 0) {
      // Allocate the index of the next set of intermediate buffers to be used
      //printf("### Reader: allocate pblIdx\n");
      pebbleIdx = readerQueue->allocate(); // This blocks when no buffers available
      //printf("### Reader: pblIdx %u\n", pebbleIdx);

      const volatile uint32_t* const __restrict__ mem = (uint32_t*)(dmaBufs[dmaBufIdx] + 4);
      //bool wait{false};
      unsigned ns{8};
      //printf("### Reader::handleDMA: Wait for dmaBufs[%u] %p\n", dmaBufIdx, mem);
      while (*mem == 0) {                  // Wait for DMA completion
        if (terminate.load(cuda::std::memory_order_acquire))
          return;
        __nanosleep(ns);
        if (ns < 256)  ns *= 2;
        //if (!wait) {
        //  wait = true;
        //  printf("### Reader::handleDMA: wait T, dmaIdx %u, pblIdx %u\n", dmaBufIdx, pebbleIdx);
        //}
      }
      //if (wait)
      //  printf("### Reader::handleDMA: wait F, dmaIdx %u, pblIdx %u\n", dmaBufIdx, pebbleIdx);
      //printf("### Reader: dma[%u] %p: sz %u\n", dmaBufIdx, mem, *mem);
      auto next = (dmaBufIdx + 1) & (dmaCount - 1);      // Prepare for the next DMA buffer
      *(volatile uint32_t*)(dmaBufs[next] + 4) = 0;      // Clear the handshake space of the next DMA buffer
      *(volatile uint8_t*)(wrEnReg + next * 4) = 1;      // Enable the DMA on this dataDev
      dmaBufferIdx = next;                               // Update global memory
      //printf("### Reader: next %lu, hand shake %p, next write enable %p\n", next, dmaBufs[next] + 4, wrEnReg + next * 4);
    }

    __threadfence();                                     // Ensure global memory is updated before the following
    unsigned value = atomicInc(&blockCount1, gridDim.x); // Thread 0 signals that it is done
    isLastBlockDone1 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last block to be done
  }
  __syncthreads();   // Synchronize to ensure that each thread (all blocks) reads the correct value of isLastBlockDone
  if (isLastBlockDone1) {               // Only last block will have set isLastBlockDone true
    if (threadIdx.x == 0) {             // Thread 0 updates global memory
      indicesValid = true;              // Thread 0 of last block signals indices in global memory are valid
      blockCount1 = 0;                  // Reset for next time
    }
  }
  unsigned ns{8};
  while(!indicesValid) {                // All threads wait for indices to become valid
    if (terminate.load(cuda::std::memory_order_acquire))
      return;
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
  }
  auto pblBufIdx = pebbleIdx;           // All threads load pebble idx into a register from global memory

  // Save the DMA descriptor and TimingHeader in pinned memory
  //if (tid == 0)  printf("### Reader: dmaIdx %3u, dmaBuf %p, pblIdx %4u\n", dmaBufIdx, dmaBufs, pblBufIdx);
  auto const __restrict__ in  = (uint32_t*)dmaBufs[dmaBufIdx];
  //if (threadIdx.x == 0)  printf("### Reader: blk %3d, dmaIdx %2u, pblIdx %4u, in %p, in[1] %u, in[9:8] %08x, %08x\n",
  //                              blockIdx.x, dmaBufIdx, pblBufIdx, in, in[1], in[9], in[8]);
  //if (tid == 0)  printf("### Reader: hdrBufs %p\n", hdrBuffers);
  auto const __restrict__ hdr = hdrBuffers + pblBufIdx * hdrBufsCnt;
  //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader: blk %3d, ob %p, pblIdx %u, hdrBufsCnt %lu, hdr %p\n", blockIdx.x, hdrBuffers, pblBufIdx, hdrBufsCnt, hdr);
  constexpr auto nDscWords = sizeof(DmaDsc)/sizeof(*in);
  constexpr auto nHdrWords = nDscWords + sizeof(TimingHeader)/sizeof(*in);
  const     auto nFrmWords = frameSize/sizeof(*in);
  //if (tid == 0)  printf("### Reader: nDscWords %lu, nHdrWords %lu\n", nDscWords, nHdrWords);
  if      (tid < nDscWords)  { hdr[tid] = in[tid]; } //printf("### Reader: tid %d, hdr %08x\n", tid, hdr[tid]); }
  else if (tid < nHdrWords)  { hdr[tid] = in[nFrmWords + tid - nDscWords]; } //printf("### Reader: tid %d, i %lu, hdr %08x\n", tid, nFrmWords + tid - nDscWords, hdr[tid]); }

  // Calibrate
  //if (threadIdx.x == 0 && blockIdx.x == 88)  printf("### Reader: blk %3d, in[1] %u, th sz %lu\n", blockIdx.x, in[1], sizeof(TimingHeader));
  auto const leaderSize = frameSize + sizeof(TimingHeader);
  if (in[1] > leaderSize) { // Calibrate only when there's a payload
    auto const __restrict__ raw = (uint16_t*)&in[leaderSize];
    //if (tid == 0)  printf("### Reader: raw %p\n", raw);
    auto const __restrict__ out = &calibBuffers[pblBufIdx * calibBufsCnt];
    //if (tid == 0)  printf("### Reader: out %p\n", out);
    auto const payloadCnt = (in[1] - leaderSize)/sizeof(*raw);
    //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu\n", payloadCnt, calibBufsCnt);
    auto const elementCnt = payloadCnt > calibBufsCnt ? calibBufsCnt : payloadCnt; // @todo: Alert to truncation
    //if (tid == 0)  printf("### Reader: payloadCnt %ld, calibBufsCnt %lu, cnt %lu\n",
    //                      payloadCnt, calibBufsCnt, elementCnt);
    auto const __restrict__ ref = refBuffers ? &refBuffers[(pblBufIdx % refBufCnt) * calibBufsCnt] : (float*)0;
    //if (tid == 0)  printf("### Reader: idx %u, refBuffers %p, ref %p\n", pblBufIdx%refBufCnt, refBuffers, ref);

    //auto const stride  = blockDim.x * gridDim.x;
    //if (tid == 0)  printf("### calibrate: nElements %lu, stride %u, idx %u, calib %p:%p\n",
    //                       elementCnt, stride, pblBufIdx, &out[0],&out[elementCnt]);

    _calibrate(out, raw, elementCnt, rangeOffset, rangeBits, pedArray, gainArray, ref);

    __syncthreads();    // Wait for all threads to complete so that thread 0 can't increment blockCount before they're done
    if (threadIdx.x == 0) {
      //if (blockIdx.x == 88)  printf("### Reader: blkIdx %d, pblIdx %u\n", blockIdx.x, pblBufIdx);

      __threadfence();                                     // Ensure global memory is updated before the following
      unsigned value = atomicInc(&blockCount2, gridDim.x); // Thread 0 signals that it is done
      isLastBlockDone2 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last  block to be done
      //printf("### Reader: blkIdx %3d, pblIdx %4u, value %3u, last %d\n", blockIdx.x, pblBufIdx, value, isLastBlockDone2);
      //if (isLastBlockDone2)  printf("### Reader: lastBlkDone T, blk %3u\n", blockIdx.x);
      //if (blockIdx.x == 0)  printf("### Reader: pblIdx %4u, gridDim %d, blkIdx %3d, value %3u, last %d\n", pblBufIdx, gridDim.x, blockIdx.x, value, isLastBlockDone2);
    }
    __syncthreads();    // Synchronize to make sure that each thread reads the correct value of isLastBlockDone
    //if (tid == 0)  printf("### Reader: pblIdx %4u, blkCnt %u\n", pblBufIdx, blockCount2);
    if (isLastBlockDone2) {
      //if (tid == 0)  printf("### Reader: lastBlkDone T, blk %3u, pblIdx %4u\n", blockIdx.x, pblBufIdx);
      if (threadIdx.x == 0) {
        //printf("### Reader: blk %3d, posting %u\n", blockIdx.x, pblBufIdx);
        readerQueue->post(pblBufIdx);
        //printf("### Reader: blk %3d, posted  %u\n", blockIdx.x, pblBufIdx);

        blockCount2 = 0;                // Reset for next time
        indicesValid = false;           // syncthreads() happens after DMA on next launch

        // Relaunch the graph
        //printf("### Reader: 2 relaunch\n");
        cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
      }
    }
  } else {                              // Transitions
    if (tid == 0) {
      //printf("### Reader: posting %u\n", pblBufIdx);
      readerQueue->post(pblBufIdx);
      //printf("### Reader: posted  %u\n", pblBufIdx);

      indicesValid = false;             // syncthreads() happens after DMA on next launch

      // Relaunch the graph
      //printf("### Reader: 1 relaunch\n");
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
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

  auto panel                = m_pool.panel();
  auto stream               = m_stream;
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
  const auto tpMP{prop.maxThreadsPerMultiProcessor};
  unsigned nMPs;
  switch (tpMP) {
    case 1536:  nMPs = 4;  break;
    case 2048:  nMPs = 2;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpMP);
      abort();
  };
  // Slightly better times seem to be achieved when nPixels/stride is an integer
  // Adjusting nBlocks for this might lead to a partially used SM, but aim for
  // maximum occupancy of the SMs
  const auto maxBpMP{prop.maxBlocksPerMultiProcessor};
  auto nThreads{tpMP/maxBpMP}; //{32}; // @todo: Move to green contexts for improved robustness
  auto nBlocks{nMPs*maxBpMP}; //{63}; //{189}; //{nMPs * tpMP / nThreads}; // {189};
  auto stride{nBlocks * nThreads};
  printf("*** Reader: blocks %u * threads %u = %u threads\n", nBlocks, nThreads, stride);

  logging::info("GPU threads per SM: %d, total threads: %u, SMs %.1f, elements per thread: %.1f\n",
                tpMP, stride, float(stride) / tpMP, float(calibBufsCnt) / stride);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream begin-capture failed")) {
    return 0;
  }

  _handleDMA<<<nBlocks, nThreads, 0, m_stream>>>(wrEnReg_d,
                                                 dmaBuffers_d,
                                                 dmaCount,
                                                 frameSize,
                                                 hostWrtBufs_d,
                                                 hostWrtBufsCnt,
                                                 calibBuffers_d,
                                                 calibBufsCnt,
                                                 m_readerQueue.d,
                                                 pedArray_d,
                                                 gainArray_d,
                                                 rangeOffset,
                                                 rangeBits,
                                                 refBuffers_d,
                                                 refBufCnt,
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

  auto const panel = m_pool.panel();
  if (panel->name != "/dev/null") {     // Else, Simulator mode
    // Ensure that timing messages are DMAed to the GPU
    dmaTgtSet(panel->coreRegs, DmaTgt_t::TGT_GPU);

    // Ensure that the DMA round-robin index starts with buffer 0
    dmaIdxReset(panel->coreRegs);
  }

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
  unsigned instance{0};
  /****************************************************************************
   * Clear the handshake space
   * Originally was cuStreamWriteValue32, but the stream functions are not
   * supported within graphs. cuMemsetD32Async acts as a good replacement.
   ****************************************************************************/
  const auto dmaBufs = panel->dmaBuffers[instance];
  chkError(cudaMemsetAsync(dmaBufs + 4, 0, sizeof(uint32_t), m_stream));
  printf("*** Reader: instance %u, dmaBuffer %p\n", instance, (void*)dmaBufs);

#ifndef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA to trigger the write
  panel->coreRegs.returnFreeListIndex(instance);
#endif // HOST_REARMS_DMA

  // Launch the Reader graph
  printf("*** Reader: Launching graph\n");
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}
