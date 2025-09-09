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

struct ResultItems
{
  unsigned                    index;
  const Pds::Eb::ResultDgram* result;
};

class TebReceiver: public Drp::TebReceiverBase
{
public:
  TebReceiver(const Parameters&, DrpBase&, const std::atomic<bool>& terminate);
  ~TebReceiver() override;
  FileWriterBase& fileWriter() override { return *m_fileWriter; }
  SmdWriterBase& smdWriter() override { return *m_smdWriter; };
  void setup();
  void teardown();
protected:
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels) override;
  void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
private:
  void _recorder();
  void _writeDgram(XtcData::Dgram*, void* devPtr);
private:
  Pds::Eb::MebContributor&         m_mon;
  const std::atomic<bool>&         m_terminate;
  cudaStream_t                     m_stream;
  std::unique_ptr<FileWriterAsync> m_fileWriter;
  std::unique_ptr<Drp::SmdWriter>  m_smdWriter;
  unsigned                         m_worker;      // For cycling through reducers
  SPSCQueue<ResultItems>           m_recordQueue;
  std::shared_ptr<Collector>       m_collector;
  std::thread                      m_recorderThread;
  const Parameters&                m_para;
};

class PGPDrp : public DrpBase
{
public:
  PGPDrp(Parameters&, MemPoolGpu&, Detector&, ZmqContext&);
  virtual ~PGPDrp();
  std::string configure(const nlohmann::json& msg);
  unsigned unconfigure();
  void reducerConfigure(XtcData::Xtc& xtc, const void* bufEnd)
                                                  { m_reducer->configure(xtc, bufEnd); }
  void reducerStart(unsigned wkr, unsigned idx)   { m_reducer->start(wkr, idx); }
  void reducerReceive(unsigned wkr, unsigned idx) { m_reducer->receive(wkr, idx); }
  void reducerEvent(XtcData::Xtc& xtc, void* be, size_t sz) { m_reducer->event(xtc, be, sz); }
  void freeBufs(unsigned idx)                     { m_collector->freeDma(idx); } // @todo: Bad name
protected:
  void pgpFlush() override;
private:
  int _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
  void _collector();
private:
  const Parameters&          m_para;
  Detector&                  m_det;
  std::atomic<bool>          m_terminate;
  cuda::atomic<uint8_t>*     m_terminate_d;
  std::vector<Reader>        m_readers;        // One reader per panel
  std::unique_ptr<Collector> m_collector;
  std::unique_ptr<Reducer>   m_reducer;
  std::thread                m_collectorThread;
  uint64_t                   m_nNoTrDgrams;
  ReaderMetrics              m_wkrMetrics;
  CollectorMetrics           m_colMetrics;
};

  } // Gpu
} // Drp
