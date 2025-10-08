#include "LcReducer.hh"

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

class LcReducerDef : public VarDef
{
public:
  enum index { lc };

  LcReducerDef()
  {
    NameVec.push_back({"lc", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


LcReducer::LcReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_compressor(pool.calibBufsSize(), 3)
{
  if (para.verbose)  m_compressor.banner();
}

// This routine records the graph that does the data reduction
void LcReducer::recordGraph(cudaStream_t       stream,
                            const unsigned&    index,
                            float const* const calibBuffers,
                            const size_t       calibBufCnt,
                            uint8_t    * const dataBuffers,
                            const size_t       dataBufCnt)
{
  m_compressor.updateGraph(stream,
                           index,
                           (uint8_t*)calibBuffers,
                           calibBufCnt * sizeof(*calibBuffers),
                           (uint8_t*)dataBuffers,
                           dataBufCnt * sizeof(*dataBuffers));
}

void LcReducer::reduce(cudaGraphExec_t graph, cudaStream_t stream, unsigned index, size_t* dataSize)
{
  chkFatal(cudaGraphLaunch(graph, stream));

  auto maxSize = m_pool.reduceBufsReserved() + m_pool.reduceBufsSize();
  auto buffer  = &m_pool.reduceBuffers_d()[index * maxSize];
  auto pSize   = buffer - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize, sizeof(*dataSize), cudaMemcpyDeviceToHost, stream));
}

unsigned LcReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("lc", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  LcReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void LcReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** LcReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

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
  data.set_array_shape(LcReducerDef::lc, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::LcReducer(para, pool, det);
}
