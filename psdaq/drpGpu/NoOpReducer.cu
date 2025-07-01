#include "NoOpReducer.hh"

#include "GpuAsyncLib.hh"
#include "MemPool.hh"
#include "drp/drp.hh"

using namespace XtcData;
using namespace Drp::Gpu;

#if 0 // @todo: Revisit
namespace Drp {
  namespace Gpu {

#define ADD_FIELD(name,ntype,ndim)  NameVec.push_back({#name, Name::ntype, ndim})

    class NoOpReducerDef : public VarDef
    {
    public:
      enum index { raw, numfields };

      NoOpReducerDef() {
        ADD_FIELD(noOp, UINT8, 0);
      }
    } noOpReducerDef;
  } // Gpu
} // Drp
#endif


NoOpReducer::NoOpReducer(const Parameters& para, const MemPoolGpu& pool) :
  ReducerAlgo(para, pool, Alg("NoOp", 0, 0, 0)),
  _calibSize(pool.calibBufSize())
{
}

static __global__ void _noOpReduce(const unsigned&              index,
                                   float**   const __restrict__ calibBuffers,
                                   uint8_t** const __restrict__ dataBuffers,
                                   unsigned  const              count)
{
  printf("*** noOpReduce 1, &index %p\n", &index);
  printf("*** noOpReduce 1,  index %u\n", index);
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  float* __restrict__ calib = calibBuffers[index];
  float* __restrict__ data  = (float*)(dataBuffers[index]);
  printf("*** noOpReduce 2, count %u\n", count);
  for (unsigned i = offset; i < count; i += blockDim.x * gridDim.x) {
    data[i] = calib[i];
  }
  *(uint32_t*)data = 0xdeadbeef;        // @todo: temporary!
  printf("*** noOpReduce 3\n");
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t&   stream,
                              const unsigned& index,
                              float**   const calibBuffers,
                              uint8_t** const dataBuffers,
                              unsigned*       extent)
{
  printf("*** NoOpReducer::recordGraph: &index %p, calibSize %zu\n", &index, _calibSize);

  unsigned count = _calibSize / sizeof(**calibBuffers);
  //int threads = 1024;
  //int blocks  = (count + threads-1) / threads; // @todo: Limit this?
  //_noOpReduce<<<blocks, threads, 0, stream>>>(index, calibBuffers, dataBuffers, 1);
  _noOpReduce<<<1, 1, 0, stream>>>(index, calibBuffers, dataBuffers, count);
  *extent = _calibSize;
}

# if 0 // @todo: Revisit
unsigned NoOpReducer::configure(Xtc& xtc, const void* bufEnd, ConfigIter& configo)
{
  // copy the detName, detType, detId from the Config Names
  Names& configNames = configo.namesLookup()[NamesId(nodeId, ConfigNamesIndex+1)].names();

  // set up the names for L1Accept data
  // Generic panel data
 {
   Alg alg("noOp", 0, 0, 0);
   m_evtNamesId[0] = NamesId(nodeId, EventNamesIndex);
   logging::debug("Constructing panel eventNames src 0x%x", unsigned(m_evtNamesId[0]));
   Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                          configNames.detName(), alg,
                                          configNames.detType(), configNames.detId(), m_evtNamesId[0], m_para->detSegment);
   names.add(xtc, bufEnd, NoOpReduceDef);
   m_namesLookup[m_evtNamesId[0]] = NameIndex(names);
 }
}

void NoOpReducer::event(Xtc& xtc, const void* bufEnd)
{
}
#endif

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters& para, const Drp::Gpu::MemPoolGpu& pool)
{
  return new Drp::Gpu::NoOpReducer(para, pool);
}
