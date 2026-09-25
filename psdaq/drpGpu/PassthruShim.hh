#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

// Not a reducer: a shim that lets the recorder handle an event whose data was
// never reduced.
//
// In pass-through mode the Reader's per-element policy writes the detector's data
// straight into the raw block ahead of each reduce buffer's payload, uncalibrated
// and unreduced -- see Gpu::Detector::rawSize() and, for ePixUHR3x2,
// EpixUHR3x2Calib in EpixUHR3x2.cu.  No reduction kernel runs at all.
//
// The recorder nonetheless needs two things that only a ReducerAlgo supplies:
//
//   * the size of the recorded payload, which it obtains by *blocking* on
//     Reducer::receive() -- so something must complete for every event or the
//     recorder stalls for ever; and
//   * the per-event Xtc array shape, written by event() via set_array_shape().
//
// This class provides both and launches nothing.  reduce() reports the fixed raw
// size the Detector asked for and returns; the graph it records is empty.
//
// It follows that a Detector using this shim owns the *description* of the data,
// because the description here is deliberately generic: the array is a flat block
// of bytes whose meaning the Detector's own Names entry gives.  For ePixUHR3x2 in
// pass-through that is RawU16Def, a typed and shaped u16 array matching the CPU
// DRP's.
//
// @todo: Stage 3 records reduced data *and* prescaled raw in the same event, at
//        which point a real reducer runs and this shim is only for CALIB.
class PassthruShim : public ReducerAlgo
{
public:
  PassthruShim(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~PassthruShim() {}

  // No graph to launch, so nothing to capture and no device-side launch path
  bool   hasGraph()    const override { return false; }

  // The reduced payload is unused in pass-through: the recorded data is the raw
  // block, which Detector::rawSize() sizes and MemPool reserves separately.
  // Returning 0 lets Reducer's maxTrSize floor decide the payload's size, which
  // must still be large enough for a transition's Xtc.
  size_t payloadSize() const override { return 0; }

  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state,
                     unsigned*    const index,
                     float const* const calibBuffers,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers,
                     size_t       const dataBufsCnt) override;
  void     reduce   (cudaGraphExec_t,
                     cudaStream_t,
                     unsigned  index,
                     size_t*   dataSize,
                     unsigned* retCode) override;
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  size_t m_rawSize;                     // Bytes the Detector reserved for raw data
};

  } // Gpu
} // Drp
