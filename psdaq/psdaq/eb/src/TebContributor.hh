/**
 * @file
 * @brief TebContributor, which batches input datagrams and posts them to the trigger event builders (TEBs), and its CtrbBatch helper.
 */
#ifndef Pds_Eb_TebContributor_hh
#define Pds_Eb_TebContributor_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include "BatchManager.hh"
#include "EbLfClient.hh"
#include "drp/spscqueue.hh"
#include "psdaq/service/fast_monotonic_clock.hh"

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <list>


namespace Pds {
  class MetricExporter;
  class EbDgram;

  namespace Eb {

    /** Single-producer single-consumer queue of EbDgram pointers. TebContributor::pending() holds the first datagram of each posted batch, or a single datagram that bypassed the TEBs. */
    using BatchQueue = SPSCQueue<const Pds::EbDgram*>;
    /** Time point of fast_monotonic_clock. */
    using time_point_t = std::chrono::time_point<fast_monotonic_clock>;

    class EbCtrbInBase;
    /** A batch of contiguous input datagrams being built by TebContributor. */
    struct CtrbBatch
    {
      /** Start a batch at dgram: entries is 1 (0 if dgram is null), start and end are dgram, tStart is now and contractor is contractor_. */
      CtrbBatch(const Pds::EbDgram* dgram, bool contractor_);

    public:
      unsigned            entries;  ///< Number of datagrams in the batch.
      time_point_t        tStart;  ///< Time at which the batch was started; used by TebContributor::timeout() and for the batch age metric.
      const Pds::EbDgram* start;  ///< First datagram of the batch, or nullptr when no batch is open.
      const Pds::EbDgram* end;  ///< Last datagram of the batch.
      bool                contractor;  ///< True if any datagram in the batch is in a readout group listed in TebCtrbParams::contractor (provides trigger input).
    };

    /** Builds batches of input datagrams in a BatchManager region and posts them over libfabric links to the TEBs. A receiver thread runs EbCtrbInBase::receiver() for the results. */
    class TebContributor
    {
    public:
      /** Keep a reference to prms, create the libfabric client and size the pending queue for numBuffers entries. */
      TebContributor(const TebCtrbParams&, unsigned numBuffers);
      /** Call shutdown(). */
      ~TebContributor();
    public:
      /** Reset the event and batch counters; returns 0. */
      int         resetCounters();
      /** Register metrics with exporter (if non-null), size the link list from prms.addrs, count the builders in prms.builders and connect to all TEBs with linksConnect(). Returns 0 or the first non-zero error code. */
      int         connect(const std::shared_ptr<MetricExporter> exporter);
      /** Drain and restart the pending queue, reinitialize the BatchManager for maxInputSize and maxEntries (aborts if maxEntries does not divide the pending queue size evenly), exchange memory-region information with the TEBs via linksConfigure() and refill each link's list of TEB_TR_BUFFERS transition buffer indices. Returns 0 or the linksConfigure() error. */
      int         configure();
      /** If running, clear the running flag, join the receiver thread, and dump and shut down the BatchManager and the pending queue; otherwise does nothing. */
      void        unconfigure();
      /** Call unconfigure(), disconnect and drop all links, and set the ID to -1. */
      void        disconnect();
      /** Clear the current batch, reset the counters of this object and of in, set the running flag and start a thread that runs in.receiver(*this, running flag). */
      void        startup(EbCtrbInBase&);
      /** Call disconnect() (harmless if already done). */
      void        shutdown();
    public:
      /**
       * Handle one input datagram. If its readout groups include bit prms.partition (the common group, per the code comment), it is appended to the open batch; the batch is posted when it has expired or the datagram is a transition other than SlowUpdate, and non-event datagrams of contractor groups are also sent to the TEBs that did not get the batch. Otherwise the open batch is posted and the datagram is marked EOL and pushed straight onto the pending queue.
       * Aborts if the pending queue overflows. Must not be called concurrently with timeout().
       */
      void        process(const Pds::EbDgram* datagram);
      /** Return the buffer index of dgram: its byte offset from the start of the batch region divided by prms.maxInputSize. */
      unsigned    index(const Pds::EbDgram* datagram) const;
      /** Return the buffer address for index, from BatchManager::fetch(). */
      void*       fetch(unsigned index) const;
      /** Call process() on the datagram at buffer index (found with fetch()). */
      void        process(unsigned index);
      /** If at least BATCH_TIMEOUT (1 ms) has passed since the start time of the current batch, post the batch if one is open and return true; otherwise return false. Must not be called concurrently with process(). */
      bool        timeout();
    public:
      /** Return the time-ordered queue of posted batches and bypassed datagrams. */
      BatchQueue& pending()  { return _pending; }
      // For reporting the outbound Tx accounting, which says whether an Input
      // this DRP believes it sent was accepted and then failed in flight.
      const std::vector<EbLfCltLink*>& links() const  { return _links; }
    private:
      int        _setupMetrics(const std::shared_ptr<MetricExporter>);
      void       _flush();
      void       _post(const Pds::EbDgram* nonEvent);
      void       _post(const CtrbBatch& batch);
    public:
      /** List of free transition buffer indices (one list per TEB link). */
      using listU32_t = std::list<uint32_t>;
    private:
      const TebCtrbParams&      _prms;
      BatchManager              _batMan;
      EbLfClient                _transport;
      std::vector<EbLfCltLink*> _links;
      std::vector<listU32_t >   _trBuffers;
      unsigned                  _id;
      unsigned                  _numEbs;
      BatchQueue                _pending; // Time ordered list of completed batches
      CtrbBatch                 _batch;
      uint64_t                  _previousPid;
    private:
      mutable uint64_t          _eventCount;
      mutable uint64_t          _batchCount;
      mutable uint64_t          _pendingSize;
      mutable uint64_t          _latPid;
      mutable int64_t           _latency;
      mutable uint64_t          _age;
      mutable uint64_t          _entries;
    private:
      std::atomic<bool>         _running;
      std::thread               _rcvrThread;
    };
  };
};

inline
unsigned Pds::Eb::TebContributor::index(const Pds::EbDgram* dgram) const
{
  unsigned offset = reinterpret_cast<const char*>(dgram) -
                    static_cast<const char*>(_batMan.batchRegion());
  uint32_t idx    = offset / _prms.maxInputSize;
  return idx;
}

inline
void* Pds::Eb::TebContributor::fetch(unsigned index) const
{
  return _batMan.fetch(index);
}

inline
void Pds::Eb::TebContributor::process(unsigned index)
{
  process(static_cast<Pds::EbDgram*>(fetch(index)));
}

#endif
