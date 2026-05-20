#include "SleekReducer.hh"

#include "MemPool.hh"
#include "Detector.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;

struct sleek_domain{ static constexpr char const* name{"SleekReducer"}; };
using sleek_scoped_range = nvtx3::scoped_range_in<sleek_domain>;


namespace Drp {
  namespace Gpu {

class SleekReducerDef : public VarDef
{
public:
  enum index { sleek };

  SleekReducerDef()
  {
    NameVec.push_back({"sleek", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


SleekReducer::SleekReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo (para, pool, det),
  m_compressor(pool.calibBufsSize(), 3.0)
{
  if (para.verbose)  m_compressor.banner();
}

// This routine records the graph that does the data reduction
void SleekReducer::recordGraph(cudaStream_t       stream,
                               unsigned*    const state_d,
                               unsigned*    const index_d,
                               float const* const calibBuffers_d,
                               size_t       const calibBufsCnt,
                               uint8_t*     const dataBuffers_d,
                               size_t       const dataBufsCnt)
{
  sleek_scoped_range r{/*"SleekReducer::recordGraph"*/}; // Expose function name via NVTX

  m_compressor.updateGraph(stream,
                           state_d,
                           index_d,
                           (uint8_t*)calibBuffers_d,
                           calibBufsCnt * sizeof(*calibBuffers_d),
                           (uint8_t*)dataBuffers_d,
                           dataBufsCnt * sizeof(*dataBuffers_d));

}

void SleekReducer::reduce(cudaGraphExec_t graph,
                          cudaStream_t    stream,
                          unsigned        index,
                          size_t*         dataSize,
                          unsigned*       retCode)
{
  sleek_scoped_range r{/*"SleekpReducer::reduce"*/}; // Expose function name via NVTX

  chkFatal(cudaGraphLaunch(graph, stream));

  // Retrieve the compressed data size from the GPU
  auto maxSize = m_pool.reduceBufsReserved() + m_pool.reduceBufsSize();
  auto buffer  = &m_pool.reduceBuffers_d()[index * maxSize];
  auto pSize   = buffer - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize, sizeof(*dataSize), cudaMemcpyDeviceToHost, stream));
  *retCode = 0;                          // @todo: TBD
}

unsigned SleekReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("sleek", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  SleekReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void SleekReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** SleekReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

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
  data.set_array_shape(SleekReducerDef::sleek, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::SleekReducer(para, pool, det);
}
