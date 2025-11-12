#include "CuSZpReducer.hh"

#include "GpuAsyncLib.hh"
#include "MemPool.hh"
#include "Detector.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"

#include <chrono>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;

struct cuszp_domain{ static constexpr char const* name{"CuSZpReducer"}; };
using cuszp_scoped_range = nvtx3::scoped_range_in<cuszp_domain>;


namespace Drp {
  namespace Gpu {

class CuSZpReducerDef : public VarDef
{
public:
  enum index { cuSZp };

  CuSZpReducerDef()
  {
    NameVec.push_back({"cuSZp", Name::UINT8, 1});
  }
};

  } // Gpu
} // Drp


CuSZpReducer::CuSZpReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo (para, pool, det),
  m_errorBound(1.2e-4)                  // Set absolute error bound
{
}

// This routine records the graph that does the data reduction
void CuSZpReducer::recordGraph(cudaStream_t       stream,
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

void CuSZpReducer::reduce(cudaGraphExec_t graph, cudaStream_t stream, unsigned index, size_t* dataSize)
{
  cuszp_scoped_range r{/*"CuSZpReducer::reduce"*/}; // Expose function name via NVTX

  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  auto dataBufsCnt  = (dataBufsRsvd + dataBufsSz) / sizeof(*dataBuffers);

  auto calibBuffer = &calibBuffers[index * calibBufsCnt];
  auto dataBuffer  = &dataBuffers[index * dataBufsCnt];

  size_t cmpSize1 = 0;
  //auto t0{Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC)};
  cuSZp_compress_plain_f32(calibBuffer, dataBuffer, calibBufsCnt, &cmpSize1, m_errorBound, stream);

  //cudaMemcpy((void*)&((size_t*)dataBuffer)[-1], &cmpSize1, sizeof(cmpSize1), cudaMemcpyHostToDevice);
  *dataSize = cmpSize1;
  //auto now{Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC)};

  //using us_t = std::chrono::microseconds;
  //auto dt{std::chrono::duration_cast<us_t>(now - t0).count()};
  //auto ratio{double(calibBufsSz)/double(cmpSize1)};
  //printf("*** dt %lu us, in %zu / out %zu = %f\n", dt, calibBufsSz, cmpSize1, ratio);
}

unsigned CuSZpReducer::configure(Xtc& xtc, const void* bufEnd)
{
  // Set up the names for L1Accept data
  Alg alg("cuSZp", 0, 0, 0);
  NamesId namesId(m_det.nodeId, ReducerNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para.detName.c_str(), alg,
                                         m_para.detType.c_str(), m_para.serNo.c_str(), namesId, m_para.detSegment);
  CuSZpReducerDef reducerDef;
  names.add(xtc, bufEnd, reducerDef);
  m_det.namesLookup()[namesId] = NameIndex(names);

 return 0;
}

void CuSZpReducer::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The Xtc header is constructed in the CPU's pebble buffer, but this buffer
  // is not used to hold all of the data.  However, bufEnd has to point to a
  // location that makes it appear that the buffer is large enough to contain
  // both the header and data so that the Xtc allocate in data.set_array_shape()
  // can succeed.  This may be larger than the pebble buffer and we therefore
  // must be careful not to write beyond its end.
  //printf("*** CuSZpReducer event: xtc %p, extent %u, size %u", &xtc, xtc.extent, dataSize);

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
  data.set_array_shape(CuSZpReducerDef::cuSZp, dataShape);
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::CuSZpReducer(para, pool, det);
}
