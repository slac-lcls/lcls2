#include "PassthruShim.hh"

#include "Detector.hh"
#include "MemPool.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;

struct pass_domain{ static constexpr char const* name{"PassthruShim"}; };
using pass_scoped_range = nvtx3::scoped_range_in<pass_domain>;


// The whole of this shim's device work: no data is moved, because the Reader's
// per-element policy already wrote the detector's data into the raw block ahead of
// this buffer's payload.  All that remains is what the surrounding machinery needs:
//
//   * the recorded size, in the word just below what was written -- below the raw
//     block here, where Reducer's _reducerLoop looks for it.  A reducer with no raw
//     block writes it just below the payload instead; either way the slot lies in
//     space the recorder only overwrites later, when it copies the Dgram, by which
//     time the size has been read.
//   * the state advance to 2, which is what tells _reducerLoop there is a result to
//     post.  Without it the recorder blocks on Reducer::receive() for ever.
//
// One thread suffices, so this is launched <<<1, 1>>>.
static __global__
void _passthru(unsigned*       const __restrict__ state,
               unsigned  const* const __restrict__ index,
               uint8_t*        const __restrict__ dataBuffers,
               size_t          const              dataBufsCnt,
               size_t          const              rawSize,
               unsigned*       const __restrict__ retCode)
{
  if (*state == 1) {
    auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];

    // The size of the data to record.  Fixed: the raw block is sized at Configure
    // by Detector::rawSize() and a short or withheld ASIC leaves zeros rather than
    // shrinking it, so unlike a real reducer there is nothing to measure.
    //
    // Below the raw block, not below the payload: the raw block holds pixel data
    // right up to the payload, so the usual slot would be clobbered by it.
    size_t* const __restrict__ extent = &((size_t*)(data - rawSize))[-1];
    *extent  = rawSize;
    *retCode = 0;

    __threadfence();                    // Publish the size before the state change

    *state = 2;                         // Tell _reducerLoop to post the result
  }
}


PassthruShim::PassthruShim(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_rawSize  (det.rawSize()),
  m_retCode_d(nullptr)
{
  // Without a raw block there is nothing for this shim to report the size of, and
  // the recorder would write a zero-length payload for every event.  Reaching here
  // means the CALIB alias asked this detector for raw recording and it has no raw
  // mode to offer -- it does not override rawSize() -- so the run would silently
  // record nothing.  Not something to limp along with.
  if (m_rawSize == 0) {
    logging::critical("PassthruShim: detector %s reserved no raw space "
                      "(Gpu::Detector::rawSize() == 0), so there is nothing to record.  "
                      "It has no raw mode, so it cannot serve a CALIB configuration.",
                      para.detName.c_str());
    abort();
  }
  chkError(cudaMalloc(&m_retCode_d,    sizeof(*m_retCode_d)));
  chkError(cudaMemset( m_retCode_d, 0, sizeof(*m_retCode_d)));

  logging::warning("PassthruShim: recording %zu B of raw data per event, unreduced",
                   m_rawSize);
}

PassthruShim::~PassthruShim()
{
  if (m_retCode_d)  cudaFree(m_retCode_d);
}

bool PassthruShim::hasGraph() const { return true; }

void PassthruShim::recordGraph(cudaStream_t       stream,
                               unsigned*    const state_d,
                               unsigned*    const index_d,
                               float const* const calibBuffers_d,
                               size_t       const calibBufsCnt,
                               uint8_t*     const dataBuffers_d,
                               size_t       const dataBufsCnt)
{
  // calibBuffers_d is deliberately unused: pass-through bypasses the calibrated
  // buffer entirely, so nothing populates it in this mode.  Anything downstream
  // that reads it -- a TrgInpGen, say -- would be reading a stale or uninitialised
  // buffer, which is why the trigger input must be data-independent here.
  _passthru<<<1, 1, 0, stream>>>(state_d,
                                 index_d,
                                 dataBuffers_d,
                                 dataBufsCnt,
                                 m_rawSize,
                                 m_retCode_d);
  chkError(cudaGetLastError(), "Launch of _passthru kernel failed");
}

void PassthruShim::reduce(cudaGraphExec_t graph,
                          cudaStream_t    stream,
                          unsigned        index,
                          size_t*         dataSize,
                          unsigned*       retCode)
{
  pass_scoped_range r{/*"PassthruShim::reduce"*/}; // Expose function name via NVTX

  // Only reached by the host-launched path, which this shim does not support.  In
  // the graph case the graph is launched once by Reducer::startup() and relaunches
  // itself from _reducerLoop, so nothing calls this.
  logging::critical("PassthruShim::reduce: unreachable -- hasGraph() is true, so the "
                    "graph relaunches itself and the host does not drive it");
  abort();
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

// No createReducer() factory here: unlike the reducers, this shim is linked into
// drp_gpu and constructed directly by Reducer::_setupAlgo().
