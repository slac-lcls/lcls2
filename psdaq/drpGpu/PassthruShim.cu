#include "PassthruShim.hh"

#include "Detector.hh"
#include "MemPool.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;


PassthruShim::PassthruShim(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_rawSize  (det.rawSize())
{
  // Without a raw block there is nothing for this shim to report the size of, and
  // the recorder would write a zero-length payload for every event.  That is a
  // configuration error -- some reducer was wanted -- not something to limp along
  // with, since the run would silently record nothing.
  if (m_rawSize == 0) {
    logging::critical("PassthruShim: detector %s reserved no raw space "
                      "(Gpu::Detector::rawSize() == 0), so there is nothing to record.  "
                      "Either enable the detector's pass-through mode or choose a reducer.",
                      para.detName.c_str());
    abort();
  }
  logging::warning("PassthruShim: recording %zu B of raw data per event, unreduced",
                   m_rawSize);
}

void PassthruShim::recordGraph(cudaStream_t       stream,
                               unsigned*    const state,
                               unsigned*    const index,
                               float const* const calibBuffers,
                               size_t       const calibBufsCnt,
                               uint8_t*     const dataBuffers,
                               size_t       const dataBufsCnt)
{
  // Deliberately empty: the data is already where it needs to be, having been
  // written to the raw block by the Reader's per-element policy.  hasGraph() is
  // false, so this is never called, but the interface requires it.
}

void PassthruShim::reduce(cudaGraphExec_t,
                          cudaStream_t,
                          unsigned  index,
                          size_t*   dataSize,
                          unsigned* retCode)
{
  // Nothing to launch and nothing to wait for.  Report the fixed size of the raw
  // block so the recorder can size its write, and a success code.
  //
  // Unlike a real reducer this does not read the size back from the GPU: the raw
  // block's size is decided at Configure by Detector::rawSize() and cannot vary
  // per event.  A short or withheld ASIC leaves zeros in place rather than
  // shrinking the array, matching the CPU DRP, so the size is genuinely constant.
  *dataSize = m_rawSize;
  *retCode  = 0;
}

int PassthruShim::configure(const nlohmann::json& configureMsg,
                            const nlohmann::json& connectMsg,
                            size_t                collectionId)
{
  return 0;                             // No configDb parameters of its own
}

unsigned PassthruShim::configure(Xtc& xtc, const void* bufEnd)
{
  logging::info("PassthruShim::configure(xtc, bufEnd)");

  // No Names of its own, deliberately.  The recorded data is the detector's raw
  // array, which the Detector already described under EventNamesIndex during its
  // own configure() -- typed and shaped, so that offline sees the same array the
  // CPU DRP writes.  Declaring a second, generic description here would leave two
  // candidate descriptions of one payload.
  return 0;
}

void PassthruShim::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // Attach the payload to the *Detector's* event Names, not to a Reducer's: in
  // pass-through the recorded bytes are the detector's raw array.
  //
  // The Xtc header is built in the CPU's pebble buffer while the payload itself is
  // on the GPU, so bufEnd is set up by the caller to make the allocate below
  // succeed even though the pebble buffer is smaller than header plus data.  See
  // the same comment in the real reducers.
  NamesId namesId(m_det.nodeId, EventNamesIndex);

  CreateData data(xtc, bufEnd, m_det.namesLookup(), namesId);

  // The shape is the detector's, so ask it rather than deriving one here: only the
  // Detector knows how its raw block is laid out.  dataSize is m_rawSize, which is
  // that layout's total extent.
  unsigned dataShape[MaxRank] = { 0 };
  auto rank = m_det.rawShape(dataShape);
  if (rank == 0) {                      // Detector offered none: fall back to flat
    dataShape[0] = dataSize;
  }
  data.set_array_shape(0, dataShape);   // Index 0: the Detector's sole raw array
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters&      para,
                                                const Drp::Gpu::MemPoolGpu& pool,
                                                Drp::Gpu::Detector&         det)
{
  return new Drp::Gpu::PassthruShim(para, pool, det);
}
