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
    Pds::EbDgram*   _binDgram  (Pds::EbDgram* dg, const Pds::Eb::CubeResultDgram&);
    void            _monitorDgram(unsigned index, const Pds::Eb::CubeResultDgram& result);
    void            _recordDgram (unsigned index, const Pds::Eb::CubeResultDgram& result);
private:
    Detector&                         m_det;
    std::vector<Pds::Eb::CubeResultDgram>  m_result;
    unsigned                          m_current;
    unsigned                          m_last; // index from CubeTebReceiver
    unsigned                          m_nbins;
    std::vector<std::atomic<bool> >   m_data_init; // reset flag for the cube
    std::vector<char*>                m_bin_data;  // one big xtc buffer per worker
    XtcData::NamesLookup              m_namesLookup;
    char*                             m_buffer;             // buffer for recording dgram
    std::vector<std::thread>          m_workerThreads;
    std::vector<SPSCQueue<unsigned> > m_workerInputQueues;  // events for each worker
    std::vector<SPSCQueue<unsigned> > m_workerOutputQueues; //
    std::thread                       m_collectorThread;    // process returns
    std::vector<Pds::Semaphore>       m_sem;
    std::atomic<bool>                 m_terminate;
};

}
