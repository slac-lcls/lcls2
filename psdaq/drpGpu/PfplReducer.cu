#include "PfplReducer.hh"

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
  ReducerAlgo (para, pool, det),
  m_compressor(pool.calibBufsSize(), 3.),
  m_error_d   (nullptr)                 // Unused as yet
{
  if (para.verbose)  m_compressor.banner();
}

static __global__
void _receive(unsigned*                const __restrict__ index,
              RingQueueHtoD<unsigned>* const __restrict__ inputQueue,
              unsigned*                const __restrict__ done)
{
  //printf("### Reducer receive: 1, done %u\n", *done);
  *done |= !inputQueue->pop(index);
  //printf("### Reducer receive: 2, idx %u, done %u\n", *index, *done);
}

/** This will re-launch the current graph */
static __global__
void _graphLoop(unsigned const*              const __restrict__ index,
                uint8_t*                     const __restrict__ dataBuffers,
                size_t                       const              dataBufsCnt,
                RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
                unsigned*                    const __restrict__ done)
{
  auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
  auto dataSize = ((size_t*)data)[-1];
  //printf("### Reducer graphLoop: push {%u, %lu}, done %u\n", *index, dataSize, *done);
  *done |= !outputQueue->push({*index, dataSize});
  if (!*done) {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
  //printf("### Reducer graphLoop: idx %u, done %u\n", *index, *done);
}

// This routine records the graph that does the data reduction
void PfplReducer::recordGraph(cudaStream_t                       stream,
                              unsigned*                    const index,
                              RingQueueHtoD<unsigned>*     const inputQueue,
                              float const*                 const calibBuffers,
                              size_t                       const calibBufsCnt,
                              uint8_t*                     const dataBuffers,
                              size_t                       const dataBufsCnt,
                              RingQueueDtoH<ReducerTuple>* const outputQueue,
                              uint64_t*                    const state_d,
                              unsigned*                    const done)
{
  pfpl_scoped_range r{/*"PfplReducer::recordGraph"*/}; // Expose function name via NVTX

  // @todo: More work is needed here
  logging::critical("PfplReducer::recordGraph: To be implemented");
  abort();

  // Handle messages from TebReceiver to process an event
  _receive<<<1, 1, 0, stream>>>(index, inputQueue, done);

  m_compressor.updateGraph(stream,
                           *index,
                           (uint8_t*)calibBuffers,
                           calibBufsCnt * sizeof(*calibBuffers),
                           (uint8_t*)dataBuffers,
                           dataBufsCnt * sizeof(*dataBuffers));

  // Re-launch! Additional behavior can be put in graphLoop as needed.
  _graphLoop<<<1, 1, 0, stream>>>(index,
                                  dataBuffers,
                                  dataBufsCnt,
                                  outputQueue,
                                  done);
}

void PfplReducer::reduce(cudaGraphExec_t graph,
                         cudaStream_t    stream,
                         unsigned        index,
                         size_t*         dataSize,
                         unsigned*       error)
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
