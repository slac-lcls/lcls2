#include "DrpBase.hh"                   // Contains base class for TebReceiver

//#define USE_TEB_COLLECTOR

namespace Pds {
  namespace Eb {
    class CubeResultDgram;
  }
}

namespace Drp {

#ifdef USE_TEB_COLLECTOR
class CubeCollector
{
public:
    CubeCollector(const Parameters&);
public:
    void queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result);
private:
    void _wait_for_complete(const Pds::Eb::CubeResultDgram& result);
    void _writeDgram(XtcData::Dgram*);
private:
    const Parameters&    m_para;
    MemPool&             m_pool;
    unsigned             m_current;
    unsigned             m_last; // index from CubeTebReceiver
    BufferedFileWriterMT     m_fileWriter;
    SmdWriter                m_smdWriter;
    SPSCQueue<void*>         m_workerQueues[];    // queue dgram for summation
    SPSCQueue<void*>         m_worker_payload[];  // intermediate summation
    Pds::Semaphore           m_worker_sem[];      // serialize recording, summation
};
#endif

class CubeTebReceiver: public TebReceiverBase
{
public:
    CubeTebReceiver(const Parameters&, DrpBase&);
#ifdef USE_TEB_COLLECTOR
    virtual FileWriterBase& fileWriter() override { return m_collector.fileWriter(); }
    virtual SmdWriterBase& smdWriter() override { return m_collector.smdWriter(); };
#else
    virtual FileWriterBase& fileWriter() override { return m_fileWriter; }
    virtual SmdWriterBase& smdWriter() override { return m_smdWriter; };
private:
    void _writeDgram(XtcData::Dgram*);
#endif
protected:
    virtual int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                             std::map<std::string, std::string>& labels) override;
    virtual void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
private:
    Pds::Eb::MebContributor& m_mon;
    const Parameters&        m_para;
#ifdef USE_TEB_COLLECTOR
    Drp::CubeCollector       m_collector;
#else
    BufferedFileWriterMT     m_fileWriter;
    SmdWriter                m_smdWriter;
#endif
};

}
