#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

// Not a reducer: in pass-through (CALIB) the Reader's policy has already written the
// raw block, and this only completes each event, which the recorder blocks on, with
// the raw size.  The Detector describes the data.  Linked into drp_gpu, not loaded,
// so CALIB can select it without the .cnf.py naming it.
class PassthruShim : public ReducerAlgo
{
public:
  PassthruShim(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~PassthruShim() {}

  // The reduced payload is unused in pass-through: the recorded data is the raw
  // block, which Detector::rawSize() sizes and MemPool reserves separately.
  size_t payloadSize() const override { return 0; }

  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state,
                     unsigned*    const index,
                     float const* const calibBuffers,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers,
                     size_t       const dataBufsCnt,
                     unsigned*    const retCode) override;
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  size_t    m_rawSize;                  // Bytes the Detector reserved for raw data
};

  } // Gpu
} // Drp
