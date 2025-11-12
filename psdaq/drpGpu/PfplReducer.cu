#include "PfplReducer.hh"

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

struct pfpl_domain{ static constexpr char const* name{"PfplReducer"}; };
using pfpl_scoped_range = nvtx3::scoped_range_in<pfpl_domain>;


namespace Drp {
  namespace Gpu {

class PfplReducerDef : public VarDef
{
public:
  enum index { pfpl };

  PfplReducerDef()
  {
    NameVec.push_back({"pfpl", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


PfplReducer::PfplReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_compressor(pool.calibBufsSize(), 3.)
{
  if (para.verbose)  m_compressor.banner();
}

// This routine records the graph that does the data reduction
void PfplReducer::recordGraph(cudaStream_t       stream,
                              const unsigned&    index,
                              float const* const calibBuffers,
                              const size_t       calibBufCnt,
                              uint8_t    * const dataBuffers,
                              const size_t       dataBufCnt)
{
  pfpl_scoped_range r{/*"PfplReducer::recordGraph"*/}; // Expose function name via NVTX

  m_compressor.updateGraph(stream,
                           index,
                           (uint8_t*)calibBuffers,
                           calibBufCnt * sizeof(*calibBuffers),
                           (uint8_t*)dataBuffers,
                           dataBufCnt * sizeof(*dataBuffers));
}

void PfplReducer::reduce(cudaGraphExec_t graph, cudaStream_t stream, unsigned index, size_t* dataSize)
{
  pfpl_scoped_range r{/*"PfplpReducer::reduce"*/}; // Expose function name via NVTX

  chkFatal(cudaGraphLaunch(graph, stream));

  // Retrieve the compressed data size from the GPU
  auto maxSize = m_pool.reduceBufsReserved() + m_pool.reduceBufsSize();
  auto buffer  = &m_pool.reduceBuffers_d()[index * maxSize];
  auto pSize   = buffer - sizeof(*dataSize);
  chkError(cudaMemcpyAsync((void*)dataSize, pSize, sizeof(*dataSize), cudaMemcpyDeviceToHost, stream));
}

unsigned PfplReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("pfpl", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  PfplReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void PfplReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** PfplReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

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
  data.set_array_shape(PfplReducerDef::pfpl, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::PfplReducer(para, pool, det);
}
