#include "LcReducer.hh"

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
  ReducerAlgo (para, pool, det),
  m_compressor(pool.calibBufsSize(), 3),
  m_error_d   (nullptr)                 // Unused as yet
{
  if (para.verbose)  m_compressor.banner();
}

/** This kernel receives a message from TebReceiver that indicates which
 * calibBuffer is ready for reducing.
 */
static __global__
void _receive(unsigned*                const __restrict__ index,
              RingQueueHtoD<unsigned>* const __restrict__ inputQueue,
              cuda::std::atomic<unsigned>  const&         terminate)
{
  //printf("### Reducer receive: 1, done %u\n", *done);
  //bool wait{false};
  unsigned ns{8};
  while (!inputQueue->pop(index)) {
    if (terminate.load(cuda::std::memory_order_acquire)) {
      printf("### Reducer receive: inputQueue empty\n");
      return;
    }
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
    //if (!wait) {
    //  wait = true;
    //  printf("### Reducer receive: wait T, tail %d, head %d\n", inputQueue->tail(), inputQueue->head());
    //}
  }
  //if (wait)
  //  printf("### Reducer receive: wait F, tail %d, head %d\n", inputQueue->tail(), inputQueue->head());
  //printf("### Reducer receive: 2, idx %u, done %u\n", *index, *done);
}

/** This will re-launch the current graph */
static __global__
void _graphLoop(unsigned const*              const __restrict__ index,
                uint8_t*                     const __restrict__ dataBuffers,
                size_t                       const              dataBufsCnt,
                RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
                cuda::std::atomic<unsigned>  const&             terminate)
{
  auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
  auto dataSize = ((size_t*)data)[-1];
  //printf("### Reducer graphLoop: push {%u, %lu}, done %u\n", *index, dataSize, *done);
  //bool wait{false};
  unsigned ns{8};
  while (!outputQueue->push({*index, dataSize})) {
    if (terminate.load(cuda::std::memory_order_acquire)) {
      printf("### Reducer graphLoop: outputQueue full\n");
      return;
    }
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
    //if (!wait) {
    //  wait = true;
    //  printf("### Reducer graphLoop: wait T, next %d, tail %d\n", outputQueue->next(), outputQueue->tail());
    //}
  }
  //if (wait)
  //  printf("### Reducer graphLoop: wait F, next %d, tail %d\n", outputQueue->next(), outputQueue->tail());
  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  //printf("### Reducer graphLoop: idx %u, done %u\n", *index, *done);
}

// This routine records the graph that does the data reduction
void LcReducer::recordGraph(cudaStream_t                        stream,
                            unsigned*                    const  index,
                            RingQueueHtoD<unsigned>*     const  inputQueue,
                            float const*                 const  calibBuffers,
                            size_t                       const  calibBufsCnt,
                            uint8_t*                     const  dataBuffers,
                            size_t                       const  dataBufsCnt,
                            RingQueueDtoH<ReducerTuple>* const  outputQueue,
                            uint64_t*                    const  state_d,
                            cuda::std::atomic<unsigned>  const& terminate_d)
{
  // @todo: More work is needed here
  logging::critical("LcReducer::recordGraph: To be implemented");
  abort();

  // Handle messages from TebReceiver to process an event
  _receive<<<1, 1, 0, stream>>>(index, inputQueue, terminate_d);

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
                                  terminate_d);
}

void LcReducer::reduce(cudaGraphExec_t graph,
                       cudaStream_t    stream,
                       unsigned        index,
                       size_t*         dataSize,
                       unsigned*       error)
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
