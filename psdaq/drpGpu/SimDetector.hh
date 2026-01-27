#pragma once

#include "Detector.hh"

#include "xtcdata/xtc/TransitionId.hh"

#include <thread>
#include <chrono>
#include <cuda_runtime.h>

namespace XtcData {
  class Xtc;
  class Dgram;
}

namespace Drp {
  class Parameters;
  class MemPool;
  namespace Gpu {

    class SimDetector : public Gpu::Detector
    {
    protected:
      SimDetector(Parameters* para, MemPoolGpu* pool, unsigned len=100);
      virtual ~SimDetector();
      void shutdown() override;
      nlohmann::json connectionInfo(const nlohmann::json& msg) override;
      void connectionShutdown() override;
      void connect(const nlohmann::json&, const std::string& collectionId) override;
      void issuePhase2(XtcData::TransitionId::Value) override;
    protected:
      virtual size_t _genL1Payload(uint8_t* buffer, size_t bufSize) = 0;
    private:
      size_t _genTimingHeader(uint8_t* buffer, XtcData::TransitionId::Value tid);
      void _trigger(CUdeviceptr buffer, uint32_t dmaSize) const;
      void _eventSimulator();
    private:
      using ms_t = std::chrono::milliseconds;
      std::atomic<bool>                       m_terminate;
      cudaStream_t                            m_stream;
      unsigned                                m_readoutGroup;
      uint32_t                                m_evtCounter;
      unsigned                                m_length;
      SPSCQueue<XtcData::TransitionId::Value> m_eventQueue;
      std::thread                             m_eventThread;
      unsigned                                m_l1Delay;
      ms_t                                    m_suPeriod;
};

  } // Gpu
} // Drp
