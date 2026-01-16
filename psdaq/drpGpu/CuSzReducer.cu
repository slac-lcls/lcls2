#include "CuSzReducer.hh"

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

class CuSzReducerDef : public VarDef
{
public:
  enum index { cuSZ };

  CuSzReducerDef()
  {
    NameVec.push_back({"cuSZ", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


CuSzReducer::CuSzReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_predictor(Lorenzo),
  m_mode     (Rel),                     // set compression mode
  m_eb       (1.2e-4),                  // set error bound
  m_m        (nullptr)
{
}

CuSzReducer::~CuSzReducer()
{
  if (m_m) {
    psz_release_resource(m_m);
    m_m = nullptr;
  }
}

// This routine records the graph that does the data reduction
void CuSzReducer::recordGraph(cudaStream_t       stream,
                              const unsigned&    index,
                              float const* const calibBuffers,
                              const size_t       calibBufCnt,
                              uint8_t    * const dataBuffers,
                              const size_t       dataBufCnt)
{
  //uint8_t* d_internal_compressed{nullptr};
  //auto m = psz_create_resource_manager(F4, calibBufCnt, 1, 1, stream);
  //
  //// @todo: This isn't right since it is evaluated at record time instead of event time
  //psz_compress_float(
  //    m, {m_predictor, DEFAULT_HISTOGRAM, Huffman, NULL_CODEC, m_mode, m_eb, DEFAULT_RADIUS},
  //    &calibBuffers[index], &m_header, &dataBuffers[index], &((size_t*)dataBuffers)[-1]);
}

void CuSzReducer::reduce(cudaGraphExec_t, cudaStream_t stream, unsigned index, size_t* dataSize)
{
  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  auto dataBufsCnt  = (dataBufsRsvd + dataBufsSz) / sizeof(*dataBuffers);

  auto calibBuffer = &calibBuffers[index * calibBufsCnt];
  auto dataBuffer  = &dataBuffers[index * dataBufsCnt];

  uint8_t* d_internal_compressed{nullptr};
  size_t compressed_len{0};
  if (!m_m)  m_m = psz_create_resource_manager(F4, calibBufsCnt, 1, 1, stream);

  psz_compress_float(
      m_m, {m_predictor, DEFAULT_HISTOGRAM, Huffman, NULL_CODEC, m_mode, m_eb, DEFAULT_RADIUS},
      calibBuffer, &m_header, &d_internal_compressed, &compressed_len);
  //printf("*** CuSzReducer::reduce: calibSz %zu, compressed_len %zu\n", calibBufsSz, compressed_len);

  // INSTRUCTION: need to copy out becore releasing resource.
  cudaMemcpy(dataBuffer, d_internal_compressed, compressed_len, cudaMemcpyDeviceToDevice);
  //cudaMemcpy((void*)&((size_t*)dataBuffer)[-1], &compressed_len, sizeof(compressed_len), cudaMemcpyHostToDevice);
  *dataSize = compressed_len;
}

unsigned CuSzReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("cuSZ", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  CuSzReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void CuSzReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** CuSzReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

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
  data.set_array_shape(CuSzReducerDef::cuSZ, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::CuSzReducer(para, pool, det);
}
