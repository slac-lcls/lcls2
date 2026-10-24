/**
 * @file
 * @brief TebReceiver, the standard DRP trigger result receiver that records events to file and sends them to the monitoring event builders.
 */
#pragma once

#include "DrpBase.hh"                   // Contains base class for TebReceiver

namespace Pds {
  namespace Eb {
    class ResultDgram;
  }
}

namespace Drp {

/** TebReceiverBase that writes recorded events with a BufferedFileWriterMT and a small-data SmdWriter, and sends events to the MEBs through the MEB contributor. */
class TebReceiver: public TebReceiverBase
{
public:
    /** Construct the base, keep the MEB contributor of drp, and create the file writer and the small-data writer with buffers of the larger of the pebble buffer size and para.maxTrSize (the file writer uses direct I/O per getDioFlag(para)). */
    TebReceiver(const Parameters&, DrpBase&);
    /** Return the BufferedFileWriterMT member that records events. */
    virtual FileWriterBase& fileWriter() override { return m_fileWriter; }
    /** Return the SmdWriter member used for the small-data output. */
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
