#pragma once

#include "TebReceiver.hh"                   // Contains base class for TebReceiver
#include "psdaq/eb/CubeResultDgram.hh"

#define MAX_CUBE_BINS 256

namespace Drp {

class CubeTebReceiver: public TebReceiver
{
public:
    CubeTebReceiver(const Parameters&, DrpBase&);
public:
    void process();
protected:
    virtual void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
private:
    void            _queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result);
    XtcData::Dgram* _binDgram  (const Pds::Eb::CubeResultDgram&);
private:
    Detector&                         m_det;
    std::vector<Pds::Eb::CubeResultDgram>  m_result;
    unsigned                          m_current;
    unsigned                          m_last; // index from CubeTebReceiver
    std::vector<char*>                m_bin_data;
    std::vector< std::vector<unsigned> >    m_bin_entries;
    char*                             m_buffer;
    std::vector<std::thread>          m_workerThreads;
    std::vector<SPSCQueue<unsigned> > m_workerInputQueues;  // events for each worker
    std::vector<SPSCQueue<unsigned> > m_workerOutputQueues; //
    std::thread                       m_collectorThread;    // process returns
    std::vector<Pds::Semaphore>       m_sem;
    std::atomic<bool>                 m_terminate;
};

}
