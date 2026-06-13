#pragma once

#include "DrpBase.hh"                   // Contains base class for TebReceiver

namespace Pds {
  namespace Eb {
    class ResultDgram;
  }
}

namespace Drp {

class TebReceiver: public TebReceiverBase
{
public:
    TebReceiver(const Parameters&, DrpBase&);
    virtual FileWriterBase& fileWriter() override { return m_fileWriter; }
    virtual SmdWriterBase& smdWriter() override { return m_smdWriter; };
protected:
    virtual int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                             std::map<std::string, std::string>& labels) override;
    virtual void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
    void _writeDgram(XtcData::Dgram*);
protected:
    Pds::Eb::MebContributor& m_mon;
    BufferedFileWriterMT     m_fileWriter;
    SmdWriter                m_smdWriter;
    const Parameters&        m_para;
};

}
