#include "NoOpReducer.hh"

#include "MemPool.hh"
#include "Detector.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

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
  m_refBufCnt(0)
{
  if (para.detType == "epixuhrsim") {
    if (para.kwargs.find("sim_l1_verify") != para.kwargs.end()) {
      if (std::stoul(para.kwargs.at("sim_l1_verify"))) {
        m_refBufs_d = det.gpuDetector()->referenceBuffers();
        m_refBufCnt = det.gpuDetector()->referenceBufCnt();
      }
    }
  }
}

NoOpReducer::~NoOpReducer()
{
}

static __device__ unsigned lErrCnt       = 0;
//static __device__ unsigned lOutSz        = 0;
static __device__ unsigned blockCount    = 0;
static __shared__ bool     isLastBlockDone;

// GPU kernel for actually performing the data reduction
// In this case, the calibrated data is just copied to the output buffer
static __global__
void _reduce(unsigned* const        __restrict__ state,
             unsigned  const* const __restrict__ index,
             float     const* const __restrict__ calibBuffers,
             size_t    const                     calibBufsCnt,
             uint8_t*  const        __restrict__ dataBuffers,
             size_t    const                     dataBufsCnt,
             float     const* const __restrict__ refBuffers,
             unsigned  const                     refBufCnt,
             unsigned* const        __restrict__ retCode)
{
  if (*state == 1) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned const idx{*index};         // Dereference only once
    float const* const __restrict__ calib = &calibBuffers[idx * calibBufsCnt];
    float*       const __restrict__ data  = (float*)(&dataBuffers[idx * dataBufsCnt]);
    float const* const __restrict__ ref   = refBuffers ? &refBuffers[(idx % refBufCnt) * calibBufsCnt] : nullptr;
    for (unsigned i = tid; i < calibBufsCnt; i += stride) {
      if (ref && (calib[i] != ref[i])) {
        atomicAdd(&lErrCnt, 1);
      }
      data[i] = calib[i];
      //atomicAdd(&lOutSz, sizeof(*data)); // @todo: Not working
    }

    __syncthreads();    // Wait for all threads to complete so that thread 0 can't increment blockCount before they're done
    if (threadIdx.x == 0) {
      __threadfence();                                        // Ensure global memory is updated before the following
      unsigned value = atomicInc(&blockCount, gridDim.x - 1); // Wraps to 0 for the next launch
      isLastBlockDone = (value == (gridDim.x - 1));           // Thread 0 determines if its block is the last block to be done
    }
    if (isLastBlockDone) {
      if (threadIdx.x == 0) {
        // Place the size of the reduced data in reserved space just before the data
        size_t* const __restrict__ extent = &((size_t*)data)[-1];
        *extent = calibBufsCnt * sizeof(*data);
        //*extent  = lOutSz;              // @todo: This returns crazy values: why?
        //lOutSz   = 0;                   // Reset for next time
        *retCode = lErrCnt;             // Count of mismatched elements
        lErrCnt  = 0;                   // Reset for next time

        // Advance to the next state
        *state = 2;
      }
    }
  }
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t       stream,
                              unsigned*    const state_d,
                              unsigned*    const index_d,
                              float const* const calibBuffers_d,
                              size_t       const calibBufsCnt,
                              uint8_t*     const dataBuffers_d,
                              size_t       const dataBufsCnt,
                              unsigned*    const retCode)
{
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
  auto nThrs{tpSM/maxBpSM}; //{32}; // @todo: Find from green context
  auto nBlks{nSMs*maxBpSM}; //20*48;
  printf("*** NoOpReducer blocks %u * threads %u = %u threads\n", nBlks, nThrs, nBlks * nThrs);
  _reduce<<<nBlks, nThrs, 0, stream>>>(state_d,
                                       index_d,
                                       calibBuffers_d,
                                       calibBufsCnt,
                                       dataBuffers_d,
                                       dataBufsCnt,
                                       m_refBufs_d,
                                       m_refBufCnt,
                                       retCode);
  chkError(cudaGetLastError(), "Launch of _reduce kernel failed");
}

int NoOpReducer::configure(const nlohmann::json& configureMsg,
                           const nlohmann::json& connectMsg,
                           size_t                collectionId)
{
  logging::info("NoOpReducer::configure(configure_json, connect_json, collection_id)");

  return 0;
}

unsigned NoOpReducer::configure(Xtc& xtc, const void* bufEnd)
{
  logging::info("NoOpReducer::configure(xtc, bufEnd)");

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
