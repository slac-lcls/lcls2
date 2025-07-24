#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>
#include "drp/DrpBase.hh"
#include "drp/spscqueue.hh"
#include "Reader.hh"
#include "Collector.hh"
#include "Reducer.hh"
#include "FileWriter.hh"

class ZmqContext;

namespace XtcData {
  class Dgram;
  class Timestamp;
}
namespace Pds {
  class MetricExporter;
  namespace Eb {
    class TebContributor;
    class ResultDgram;
  }
}

namespace Drp {
  namespace Gpu {

class Detector;
class MemPoolGpu;
class RingIndexDtoD;
class RingIndexDtoH;
class RingIndexHtoD;

class TebReceiver: public Drp::TebReceiverBase
{
public:
  TebReceiver(const Parameters&, DrpBase&,
              const std::atomic<bool>& terminate_h,
              const cuda::atomic<int>& terminate_d);
  ~TebReceiver() override;
  FileWriterBase& fileWriter() override { return *m_fileWriter; }
  SmdWriterBase& smdWriter() override { return *m_smdWriter; };
  void setupReducers(std::shared_ptr<Collector>);
protected:
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels) override;
  void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
private:
  void _recorder();
  void _writeDgram(XtcData::Dgram*, void* devPtr, size_t size);
private:
  std::vector<Reducer>                   m_reducers;
  Pds::Eb::MebContributor&               m_mon;
  const std::atomic<bool>&               m_terminate_h;    // Avoid PCIe transfer of _d
  const cuda::atomic<int>&               m_terminate_d;    // Managed memory pointer
  cudaStream_t                           m_stream;
  std::unique_ptr<FileWriterAsync>       m_fileWriter;
  std::unique_ptr<Drp::SmdWriter>        m_smdWriter;
  unsigned                               m_reducer;   // For cycling through reducers
  SPSCQueue<const Pds::Eb::ResultDgram*> m_resultQueue;
  std::shared_ptr<Collector>             m_collector;
  std::thread                            m_recorderThread;
  const Parameters&                      m_para;
};

class PGPDrp : public DrpBase
{
public:
  PGPDrp(Parameters&, MemPoolGpu&, Detector&, ZmqContext&);
  virtual ~PGPDrp();
  std::string configure(const nlohmann::json& msg);
  unsigned unconfigure();
protected:
  void pgpFlush() override;
private:
  int _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
  void _collector();
private:
  const Parameters&          m_para;
  Detector&                  m_det;
  std::atomic<bool>          m_terminate_h;    // Avoid PCIe transfer of _d
  cuda::atomic<int>*         m_terminate_d;    // Managed memory pointer
  std::vector<Reader>        m_readers;        // One reader per panel
  std::shared_ptr<Collector> m_collector;
  std::thread                m_collectorThread;
  uint64_t                   m_nNoTrDgrams;
  ReaderMetrics              m_wkrMetrics;
  CollectorMetrics           m_colMetrics;
};

  } // Gpu
} // Drp
