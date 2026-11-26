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


// Moves no data: it writes the size where _reducerLoop looks for it and advances the
// state to 2, without which the recorder blocks for ever.  Launched <<<1, 1>>>.
static __global__
void _passthru(unsigned*       const __restrict__ state,
               unsigned  const* const __restrict__ index,
               uint8_t*        const __restrict__ dataBuffers,
               size_t          const              dataBufsCnt,
               size_t          const              rawSize)
{
  if (*state == 1) {
    auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];

    // Fixed size, written below the raw block, not the payload: raw data runs right
    // up to the payload and would clobber the usual slot.
    size_t* const __restrict__ extent = &((size_t*)(data - rawSize))[-1];
    *extent  = rawSize;

    __threadfence();                    // Publish the size before the state change

    *state = 2;                         // Tell _reducerLoop to post the result
  }
}


PassthruShim::PassthruShim(const Parameters& para, const MemPoolGpu& pool, Detector& det) :
  ReducerAlgo(para, pool, det),
  m_rawSize  (det.rawSize())
{
  // CALIB on a detector with no raw mode would silently record nothing
  if (m_rawSize == 0) {
    logging::critical("PassthruShim: detector %s reserved no raw space "
                      "(Gpu::Detector::rawSize() == 0), so there is nothing to record.  "
                      "It has no raw mode, so it cannot serve a CALIB configuration.",
                      para.detName.c_str());
    abort();
  }
  logging::warning("PassthruShim: recording %zu B of raw data per event, unreduced",
                   m_rawSize);
}

void PassthruShim::recordGraph(cudaStream_t       stream,
                               unsigned*    const state_d,
                               unsigned*    const index_d,
                               float const* const calibBuffers_d,
                               size_t       const calibBufsCnt,
                               uint8_t*     const dataBuffers_d,
                               size_t       const dataBufsCnt,
                               unsigned*    const retCode)
{
  // calibBuffers_d is unused: nothing populates it in pass-through
  _passthru<<<1, 1, 0, stream>>>(state_d,
                                 index_d,
                                 dataBuffers_d,
                                 dataBufsCnt,
                                 m_rawSize);
  chkError(cudaGetLastError(), "Launch of _passthru kernel failed");
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

  // No Names of its own: the Detector already described its raw array
  return 0;
}

void PassthruShim::event(Xtc& xtc, const void* bufEnd, unsigned dataSize)
{
  // The recorded bytes *are* the detector's raw array
  m_det.rawEvent(xtc, bufEnd);
}

// No createReducer() factory here: unlike the reducers, this shim is linked into
// drp_gpu and constructed directly by Reducer::_setupAlgo().
