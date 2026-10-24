/**
 * @file
 * @brief CubeTebReceiver, a TebReceiver that bins detector raw data into a cube according to the trigger results.
 */
#pragma once

#include "TebReceiver.hh"                   // Contains base class for TebReceiver
#include "CubeResult.hh"

namespace Drp {

class CubeData;

/** TebReceiver that accumulates the detector raw data (Detector::rawDef()) into bins chosen by the Cube trigger results, using nCubeWorkers worker threads; the full cube is recorded at EndRun and single bins are recorded or monitored as the results request (per the file comment). */
class CubeTebReceiver: public TebReceiver
{
public:
    /** Construct the TebReceiver base and the per-buffer result copies, with one initialization flag and one semaphore per cube worker. */
    CubeTebReceiver(const Parameters&, DrpBase&);
public:
    /** Thread body (named drp/CubeTebProc): for each completed result in order, take the matching index from the worker output queue (aborting on a mismatch), monitor and record the requested bins, free the transition and pebble buffers, and zero the flushed bins (or mark the whole cube for reinitialization). Busy-waits while no new result is available; exits when terminating or a worker queue is shut down. */
    void finalize();
protected:
    virtual void complete(unsigned index, const Pds::Eb::ResultDgram&) override;
private:
    void            _queueDgram(unsigned index,   const Pds::Eb::ResultDgram& result);
    Pds::EbDgram*   _binDgram  (Pds::EbDgram* dg, const std::vector<unsigned>& bins);
    void            _monitorDgram(unsigned index, const Pds::Eb::ResultDgram&, const std::vector<unsigned>& bins);
    void            _recordDgram (unsigned index, const Pds::Eb::ResultDgram&, const std::vector<unsigned>& bins);
private:
    Detector&                         m_det;
    CubeResult                        m_resultParse;
    std::vector<Pds::Eb::ResultDgram>  m_result;
    unsigned                          m_current;
    unsigned                          m_last; // index from CubeTebReceiver
    unsigned                          m_nbins;
    std::vector<std::atomic<bool> >   m_data_init; // reset flag for the cube
    std::vector<CubeData*>            m_cubedata;  // one big xtc buffer per worker
    XtcData::NamesLookup              m_namesLookup;
    char*                             m_buffer;             // buffer for recording dgram
    std::vector<std::thread>          m_workerThreads;
    std::vector<SPSCQueue<unsigned> > m_workerInputQueues;  // events for each worker
    std::vector<SPSCQueue<unsigned> > m_workerOutputQueues; //
    std::thread                       m_collectorThread;    // process returns
    std::vector<Pds::Semaphore>       m_sem;
    Pds::Semaphore                    m_flush_sem;
    std::atomic<bool>                 m_terminate;
};

}
