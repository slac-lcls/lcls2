#include "NoOpReducer.hh"

#include "GpuAsyncLib.hh"
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
  m_refBufs_d (nullptr),
  m_refBufCnt (0),
  m_errorCnt_d(nullptr)
{
  if (para.detType == "epixuhrsim") {
    if (para.kwargs.find("sim_l1_verify") != para.kwargs.end()) {
      if (std::stoul(const_cast<Parameters&>(para).kwargs["sim_l1_verify"])) {
        m_refBufs_d = det.gpuDetector()->referenceBuffers();
        m_refBufCnt = det.gpuDetector()->referenceBufCnt();
      }
    }
  }
  chkError(cudaMalloc(&m_errorCnt_d,    sizeof(*m_errorCnt_d)));
  chkError(cudaMemset( m_errorCnt_d, 0, sizeof(*m_errorCnt_d)));
}

// GPU kernel for actually performing the data reduction
// In this case, the calibrated data is just copied to the output buffer
static __global__ void _noOpReduce(unsigned const*                    index,
                                   float    const* const __restrict__ calibBuffers,
                                   size_t   const                     calibBufsCnt,
                                   uint8_t       * const __restrict__ dataBuffers,
                                   size_t   const                     dataBufsCnt,
                                   float    const* const __restrict__ refBuffers,
                                   unsigned const                     refBufCnt,
                                   unsigned*       const __restrict__ error)
{
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  unsigned const idx{*index}; // Dereference only once
  float const* const __restrict__ calib = &calibBuffers[idx * calibBufsCnt];
  float*       const __restrict__ data  = (float*)(&dataBuffers[idx * dataBufsCnt]);
  float const* const __restrict__ ref   = refBuffers ? &refBuffers[(idx % refBufCnt) * calibBufsCnt] : nullptr;
  //if (offset == 0)  printf("### NoOpReduce: calibCnt %lu, dataCnt %lu, stride %u, idx %u, calib %p:%p, data %p:%p\n",
  //                         calibBufsCnt, dataBufsCnt, stride, idx, &calib[0],&calib[calibBufsCnt], &data[0], &data[calibBufsCnt]);
  for (unsigned i = offset; i < calibBufsCnt; i += stride) {
    //if (i < 4) {
    //  printf("### NoOpReduce: idx %u, cnt %lu, ref %p: i %u, calib %f, ref %f\n", idx%refBufCnt, calibBufsCnt, ref, i, calib[i], ref ? ref[i] : 0.f);
    //}
    if (ref && (calib[i] != ref[i])) {
    //  printf("### NoOpReduce: blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n", blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
      ++(*error);
    }
    data[i] = calib[i];
  }

  // Place the size of the reduced data just before the data
  if (offset == 0) {
    size_t* const __restrict__ extent = &((size_t*)data)[-1];
    *extent = calibBufsCnt * sizeof(*calib); //Buffers);
  }
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t          stream,
                              unsigned const&       index,
                              float    const* const calibBuffers,
                              size_t   const        calibBufsCnt,
                              uint8_t       * const dataBuffers,
                              size_t   const        dataBufsCnt)
{
  printf("*** calibBufsCnt %zu, dataBufsCnt %zu, refBufs_d %p\n", calibBufsCnt, dataBufsCnt, m_refBufs_d);
  int threads = 32; //1024;
  int blocks  = 20*48; //(calibBufsCnt + threads-1) / threads; // @todo: Limit this?
  _noOpReduce<<<blocks, threads, 0, stream>>>(&index,
                                              calibBuffers,
                                              calibBufsCnt,
                                              dataBuffers,
                                              dataBufsCnt,
                                              m_refBufs_d,
                                              m_refBufCnt,
                                              m_errorCnt_d);
}

#if 1 // hasGraph == true case
bool NoOpReducer::hasGraph() const { return true; }

void NoOpReducer::reduce(cudaGraphExec_t graph,
                         cudaStream_t    stream,
                         unsigned        index,
                         size_t*         dataSize,
                         unsigned*       errorCnt)
{
  noop_scoped_range r{/*"NoOpReducer::reduce"*/}; // Expose function name via NVTX

  printf("*** NoOpReducer::reduce\n");

  chkFatal(cudaGraphLaunch(graph, stream));

  auto maxSize = m_pool.reduceBufsReserved() + m_pool.reduceBufsSize();
  auto buffer  = &m_pool.reduceBuffers_d()[index * maxSize];
  auto pSize   = buffer - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize,        sizeof(*dataSize),    cudaMemcpyDefault, stream));
  chkError(cudaMemcpyAsync((void*)errorCnt, m_errorCnt_d, sizeof(m_errorCnt_d), cudaMemcpyDefault, stream));
}
#else // hasGraph == false case
bool NoOpReducer::hasGraph() const { return false; }

void NoOpReducer::reduce(cudaGraphExec_t,
                         cudaStream_t stream,
                         unsigned     index,
                         size_t*      dataSize,
                         unsigned*    errorCnt)
{
  noop_scoped_range r{/*"NoOpReducer::reduce"*/}; // Expose function name via NVTX

  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  auto dataBufsCnt  = (dataBufsRsvd + dataBufsSz) / sizeof(*dataBuffers);

  int threads = 32; //1024;
  int blocks  = 20*48; //(calibBufsCnt + threads-1) / threads; // @todo: Limit this?
  printf("*** NoOp::reduce: 1 blockks %d, threads %d, grid size %d\n", blocks, threads, blocks * threads);
  _noOpReduce<<<blocks, threads, 0, stream>>>(index,
                                              calibBuffers,
                                              calibBufsCnt,
                                              dataBuffers,
                                              dataBufsCnt,
                                              m_refBufs_d,
                                              m_errorCnt_d);

  printf("*** NoOp::reduce: 2\n");

  auto maxSize = dataBufsRsvd + dataBufsSz;
  auto buffer  = &dataBuffers[index * maxSize];
  auto pSize   = buffer - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize,        sizeof(*dataSize),    cudaMemcpyDefault, stream));
  chkError(cudaMemcpyAsync((void*)errorCnt, m_errorCnt_d, sizeof(m_errorCnt_d), cudaMemcpyDefault, stream));
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
