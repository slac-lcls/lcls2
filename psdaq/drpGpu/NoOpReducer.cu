#include "NoOpReducer.hh"

#include "MemPool.hh"
#include "Detector.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

// Define the HAS_GRAPH macro to select the CUDA graph method of running the
// Reducer.  Leave it undefined to select the raw kernel launching method.
#define HAS_GRAPH

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;

struct noop_domain{ static constexpr char const* name{"NoOpReducer"}; };
using noop_scoped_range = nvtx3::scoped_range_in<noop_domain>;


namespace Drp {
  namespace Gpu {

class NoOpReducerDef : public VarDef
{
public:
  enum index { noOp };

  NoOpReducerDef()
  {
    NameVec.push_back({"noOp", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


NoOpReducer::NoOpReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_refBufs_d(nullptr),
  m_refBufCnt(0),
  m_retCode_d(nullptr)
{
  if (para.detType == "epixuhrsim") {
    if (para.kwargs.find("sim_l1_verify") != para.kwargs.end()) {
      if (std::stoul(const_cast<Parameters&>(para).kwargs["sim_l1_verify"])) {
        m_refBufs_d = det.gpuDetector()->referenceBuffers();
        m_refBufCnt = det.gpuDetector()->referenceBufCnt();
      }
    }
  }
  chkError(cudaMalloc(&m_retCode_d,    sizeof(*m_retCode_d)));
  chkError(cudaMemset( m_retCode_d, 0, sizeof(*m_retCode_d)));
}

NoOpReducer::~NoOpReducer()
{
  if (m_retCode_d)  cudaFree(m_retCode_d);
}

static __device__ unsigned lErrCnt       = 0;
//static __device__ unsigned lOutSz        = 0;
static __device__ unsigned blockCount    = 0;
static __shared__ bool     isLastBlockDone;

// GPU kernel for actually performing the data reduction
// In this case, the calibrated data is just copied to the output buffer
static __global__
void _reduce(unsigned* const        __restrict__ state,
#ifdef HAS_GRAPH
             unsigned  const* const __restrict__ index,
#else
             unsigned  const                     index,
#endif
             float     const* const __restrict__ calibBuffers,
             size_t    const                     calibBufsCnt,
             uint8_t*  const        __restrict__ dataBuffers,
             size_t    const                     dataBufsCnt,
             float     const* const __restrict__ refBuffers,
             unsigned  const                     refBufCnt,
             unsigned* const        __restrict__ retCode)
{
#ifndef HOST_LAUNCHED_REDUCERS
  if (*state == 1)
#endif
  {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#ifdef HAS_GRAPH
    unsigned const idx{*index};         // Dereference only once
#else
    unsigned const idx{index};
#endif
    float const* const __restrict__ calib = &calibBuffers[idx * calibBufsCnt];
    float*       const __restrict__ data  = (float*)(&dataBuffers[idx * dataBufsCnt]);
    //float const* const __restrict__ ref   = refBuffers ? &refBuffers[(idx % refBufCnt) * calibBufsCnt] : nullptr;
    //if (tid == 0)  printf("### NoOpReducer: calibCnt %lu, dataCnt %lu, stride %u, idx %u, calib %p:%p, data %p:%p\n",
    //                      calibBufsCnt, dataBufsCnt, stride, idx, &calib[0],&calib[calibBufsCnt], &data[0], &data[calibBufsCnt]);
    for (unsigned i = tid; i < calibBufsCnt; i += stride) {
      //if (i < 4) {
      //  if (!ref)
      //    printf("### NoOpReducer: blk %d, thr %d, idx %u, cnt %lu, i %u, calib %f\n",
      //           blockIdx.x, threadIdx.x, idx, calibBufsCnt, i, calib[i]);
      //  else
      //    printf("### NoOpReducer: blk %d, thr %d, idx %u, cnt %lu, ref %p: i %u, calib %f, ref %f\n",
      //           blockIdx.x, threadIdx.x, idx%refBufCnt, calibBufsCnt, &ref[i], i, calib[i], ref ? ref[i] : 0.f);
      //}
      //if (ref && (calib[i] != ref[i])) {
      //  printf("### NoOpReducer: blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n",
      //         blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
      //  atomicAdd(&lErrCnt, 1);
      //}
      data[i] = calib[i];
      //atomicAdd(&lOutSz, sizeof(*data)); // @todo: Not working
    }

    __syncthreads();    // Wait for all threads to complete so that thread 0 can't increment blockCount before they're done
    if (threadIdx.x == 0) {
      __threadfence();                                    // Ensure global memory is updated before the following
      unsigned value = atomicInc(&blockCount, gridDim.x); // Thread 0 signals that it is done
      isLastBlockDone = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last  block to be done
    }
    if (isLastBlockDone) {
      if (threadIdx.x == 0) {
        // Place the size of the reduced data in reserved space just before the data
        size_t* const __restrict__ extent = &((size_t*)data)[-1];
        *extent = calibBufsCnt * sizeof(*data);
        //*extent  = lOutSz;              // @todo: This returns crazy values: why?
        //lOutSz   = 0;                   // Reset for next time
        *retCode = lErrCnt != 0;
        lErrCnt  = 0;                   // Reset for next time

#ifndef HOST_LAUNCHED_REDUCERS
        // Advance to the next state
        *state = 2;
#endif
      }
    }
  }
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t       stream,
                              unsigned*    const state_d,
                              unsigned*    const index_d,
                              float const* const calibBuffers,
                              size_t       const calibBufsCnt,
                              uint8_t*     const dataBuffers,
                              size_t       const dataBufsCnt)
{
#ifdef HAS_GRAPH
  // @todo: Use green context SM splitting results here
  cudaDeviceProp prop;
  chkError(cudaGetDeviceProperties(&prop, 0));
  const auto tpSM{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;
  switch (tpSM) {
    case 1536:  nSMs = 20;  break;
    case 2048:  nSMs = 10;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpSM);
      abort();
  };
  const auto maxBpSM{prop.maxBlocksPerMultiProcessor};
  auto threads{tpSM/maxBpSM}; //{32}; // @todo: Move to green contexts for improved robustness
  auto blocks {nSMs*maxBpSM}; //20*48; //(calibBufsCnt + threads-1) / threads; // @todo: Limit this?
  printf("NoOpReducer blocks %u * threads %u = %u threads\n", blocks, threads, blocks * threads);
  _reduce<<<blocks, threads, 0, stream>>>(state_d,
                                          index_d,
                                          calibBuffers,
                                          calibBufsCnt,
                                          dataBuffers,
                                          dataBufsCnt,
                                          m_refBufs_d,
                                          m_refBufCnt,
                                          m_retCode_d);
  chkError(cudaGetLastError(), "Launch of _reduce kernel failed");
#endif
}

#ifdef HAS_GRAPH // hasGraph == true case
bool NoOpReducer::hasGraph() const { return true; }

void NoOpReducer::reduce(cudaGraphExec_t graph,
                         cudaStream_t    stream,
                         unsigned        index,
                         size_t*         dataSize,
                         unsigned*       retCode)
{
  noop_scoped_range r{/*"NoOpReducer::reduce"*/}; // Expose function name via NVTX

  //printf("*** NoOpReducer::reduce\n");

  chkFatal(cudaGraphLaunch(graph, stream));

  auto maxSize  = m_pool.reduceBufsReserved() + m_pool.reduceBufsSize();
  auto buffer_d = &m_pool.reduceBuffers_d()[index * maxSize];
  auto pSize_d  = buffer_d - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize_d,     sizeof(*dataSize), cudaMemcpyDefault, stream));
  chkError(cudaMemcpyAsync((void*)retCode,  m_retCode_d, sizeof(*retCode),  cudaMemcpyDefault, stream));
}
#else // hasGraph == false case
bool NoOpReducer::hasGraph() const { return false; }

void NoOpReducer::reduce(cudaGraphExec_t,
                         cudaStream_t stream,
                         unsigned     index,
                         size_t*      dataSize,
                         unsigned*    retCode)
{
  noop_scoped_range r{/*"NoOpReducer::reduce"*/}; // Expose function name via NVTX

  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  auto dataBufsCnt  = (dataBufsRsvd + dataBufsSz) / sizeof(*dataBuffers);

  // @todo: Use green context SM splitting results here
  cudaDeviceProp prop;
  chkError(cudaGetDeviceProperties(&prop, 0));
  const auto tpSM{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;
  switch (tpSM) {
    case 1536:  nSMs = 20;  break;
    case 2048:  nSMs = 10;  break;
    default:
      logging::critical("Unexpected number of threads per MultiProcessor %u", tpSM);
      abort();
  };
  const auto maxBpSM{prop.maxBlocksPerMultiProcessor};
  auto threads{tpSM/maxBpSM}; //{32}; // @todo: Move to green contexts for improved robustness
  auto blocks {nSMs*maxBpSM}; //20*48; //(calibBufsCnt + threads-1) / threads; // @todo: Limit this?
  printf("*** NoOp::reduce: 1 blocks %d, threads %d, grid size %d\n", blocks, threads, blocks * threads);
  _reduce<<<blocks, threads, 0, stream>>>(nullptr, // Unused state variable
                                          index,
                                          calibBuffers,
                                          calibBufsCnt,
                                          dataBuffers,
                                          dataBufsCnt,
                                          m_refBufs_d,
                                          m_refBufCnt,
                                          m_retCode_d);
  chkError(cudaGetLastError(), "Launch of _reduce kernel failed");

  printf("*** NoOp::reduce: 2\n");

  auto maxSize  = dataBufsRsvd + dataBufsSz;
  auto buffer_d = &dataBuffers[index * maxSize];
  auto pSize_d  = buffer_d - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize_d,    sizeof(*dataSize), cudaMemcpyDefault, stream));
  chkError(cudaMemcpyAsync((void*)retCode, m_retCode_d, sizeof(*retCode),  cudaMemcpyDefault, stream));
  chkError(cudaStreamSynchronize(stream));
}
#endif

unsigned NoOpReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("noOp", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  NoOpReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void NoOpReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** NoOpReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

  // Data is Reduced data
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);

  // CreateData places into the Xtc, in one contiguous block:
  // - the ShapesData Xtc
  // - the Shapes Xtc with its payload
  // - the Data Xtc (the payload of which is on the GPU)
  CreateData data(xtc, bufEnd, m_det.namesLookup(), namesId);

  // Update the header with the size and shape of the data payload.
  // This does not write beyond the Xtc header in the pebble buffer.
  unsigned dataShape[MaxRank] = { dataSize };
  data.set_array_shape(NoOpReducerDef::noOp, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::NoOpReducer(para, pool, det);
}
