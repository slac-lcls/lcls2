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
  ReducerAlgo(para, pool, det)
{
}

// GPU kernel for actually performing the data reduction
// In this case, the calibrated data is just copied to the output buffer
static __global__ void _noOpReduce(const unsigned&                 index,
                                   float const* const __restrict__ calibBuffers,
                                   const size_t                    calibBufsCnt,
                                   uint8_t    * const __restrict__ dataBuffers,
                                   const size_t                    dataBufsCnt,
                                   unsigned const                  count)
{
  //printf("### noOpReduce 1, &index %p\n", &index);
  //printf("### noOpReduce 1,  index %u\n", index);
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  float const* const __restrict__ calib = &calibBuffers[index * calibBufsCnt];
  float*       const __restrict__ data  = (float*)(&dataBuffers[index * dataBufsCnt]);
  //printf("### noOpReduce 2, count %u\n", count);
  for (unsigned i = offset; i < count; i += blockDim.x * gridDim.x) {
    data[i] = calib[i];
  }
  //printf("### noOpReduce 3\n");
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t&      stream,
                              const unsigned&    index,
                              float const* const calibBuffers,
                              const size_t       calibBufsCnt,
                              uint8_t    * const dataBuffers,
                              const size_t       dataBufsCnt,
                              unsigned*          extent)
{
  //printf("*** NoOpReducer::recordGraph: &index %p, calibSize %zu\n", &index, _calibSize);

  int threads = 1024;
  int blocks  = (calibBufsCnt + threads-1) / threads; // @todo: Limit this?
  _noOpReduce<<<blocks, threads, 0, stream>>>(index, calibBuffers, calibBufsCnt, dataBuffers, dataBufsCnt, 1);
  //_noOpReduce<<<1, 1, 0, stream>>>(index, calibBuffers, calibBufsCnt, dataBuffers, dataBufsCnt, calibBufsCnt);
  *extent = calibBufsCnt * sizeof(*calibBuffers);
}

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
