// A device-side version of drp/EventBatcher.hh
// see https://confluence.slac.stanford.edu/display/ppareg/AxiStream+Batcher+Protocol+Version+1
//
// Unlike the drp/ version, the walk here is __device__ only: the DMA buffer is
// never copied to the CPU, so there is no host-side consumer.  The only __host__
// code below manages the storage for a scan and lets the host see how one fared.
//
// Differences from drp/EventBatcher.hh, all forced by or useful for the GPU:
// - psalg::SysLog and exceptions are unavailable on the device, so protocol
//   violations are latched into an EvtBatcherStatus that the host reads out of
//   pinned memory instead of being logged or thrown;
// - EvtBatcherSubFrame records a payload-relative *offset* rather than a
//   pointer.  This is what makes a scan reusable: the sub-frame layout repeats
//   from event to event, but each event lands in a different DMA buffer, so a
//   cached pointer would dangle while a cached offset stays valid;
// - EvtBatcherSubFrames caches one scan and the payload size that produced it,
//   so the (inherently serial, single-threaded) walk happens once per
//   configuration rather than once per event, and the resulting flat,
//   tdest-indexed table is read in parallel by every thread of the grid.
//
// A note on memory ordering: the sub-frame tails live in the FPGA-written DMA
// buffer, which the graph reuses every dmaCount events.  Plain loads suffice:
// PCIe writes into device memory are visible through the L2, which is what makes
// Reader.cu's volatile poll in _waitForDMA work, and the L1 is invalidated at
// each kernel launch.  Nothing below needs to be volatile.
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "gpuUtils.hh"                  // For chkError

// Set to 1 to have the device report protocol violations via printf
#ifndef DRP_GPU_EVTBATCHER_VERBOSE
#  define DRP_GPU_EVTBATCHER_VERBOSE 0
#endif
#if DRP_GPU_EVTBATCHER_VERBOSE
#  define _EB_WARN(fmt, ...) printf("*** EventBatcher: " fmt "\n", ##__VA_ARGS__)
#else
#  define _EB_WARN(fmt, ...) do { } while (0)
#endif

namespace Drp {
  namespace Gpu {

    // A kernel can neither log nor throw, so corruption is latched and handed
    // back to the host through the pinned mirror in EvtBatcherSubFrames
    enum EvtBatcherStatus : unsigned {
      EvtBatcherOk = 0,
      EvtBatcherTooShort,                // Batch can't hold a header and one tail
      EvtBatcherZeroSize,                // A tail claims a sub-frame size of 0
      EvtBatcherCorrupt,                 // Walk ran off the front of the batch
      EvtBatcherOverflow,                // A tdest exceeded subframeCount()-1
    };

    // The one function the host needs; also makes the optional device printfs
    // legible, hence __device__ as well
    __host__ __device__
    inline const char* evtBatcherStatusName(unsigned status)
    {
      switch (status) {
        case EvtBatcherOk:       return "ok";
        case EvtBatcherTooShort: return "batch too short";
        case EvtBatcherZeroSize: return "sub-frame size is 0";
        case EvtBatcherCorrupt:  return "sub-frame walk overran the batch";
        case EvtBatcherOverflow: return "tdest exceeds the expected sub-frame count";
        default:                 return "unknown";
      }
    }

#pragma pack(push,1)
    class EvtBatcherHeader {
    public:
      enum { VERSION = 1 };
    public:
      __device__ static unsigned lineWidth(unsigned w) { return 2u << w; }
      __device__ unsigned    lineWidth() const { return lineWidth(width); }
      // The first sub-frame's data begins on the line after the header
      __device__ const void* next()      const { return (const uint8_t*)this + lineWidth(width); }
      __device__ bool        valid()     const { return version == VERSION; }
    public:
      unsigned version:4;
      unsigned width:4;
      uint8_t  sequence_count;
    };

    class EvtBatcherSubFrameTail {
    public:
      // The tail follows its sub-frame's line-padded data
      __device__ const uint8_t* data()       const { return (const uint8_t*)this - totSize(); }
      __device__ unsigned       width()      const { return _width; }
      __device__ unsigned       tdest()      const { return _tdest; }
      __device__ unsigned       tUserFirst() const { return _tuser_first; }
      __device__ unsigned       tUserLast()  const { return _tuser_last; }
      __device__ uint32_t       size()       const { return _size; }
      // The size rounded up to the nearest line boundary, which depends on the
      // "width" parameter.  Unlike drp/, a size of 0 is not diagnosed here (it
      // yields 0); EvtBatcherIterator::next() catches it.
      __device__ uint32_t       totSize()    const {
        const uint32_t lw = EvtBatcherHeader::lineWidth(_width);
        return (_size + lw - 1) & ~(lw - 1);
      }
    private:
      uint32_t _size;
      uint8_t  _tdest;
      uint8_t  _tuser_first;
      uint8_t  _tuser_last;
      uint8_t  _width;
    };
#pragma pack(pop)

    static_assert(sizeof(EvtBatcherHeader)       == 2, "EvtBatcherHeader must be 2 bytes");
    static_assert(sizeof(EvtBatcherSubFrameTail) == 8, "EvtBatcherSubFrameTail must be 8 bytes");

    class EvtBatcherIterator {
    public:
      __device__
      EvtBatcherIterator(const EvtBatcherHeader* ebh, size_t bytes) :
        // compute the first subframe tail ptr
        _lw    (ebh->lineWidth()),
        _next  ((const uint8_t*)ebh + bytes - _lw),
        _end   ((const uint8_t*)ebh->next()),
        _status(EvtBatcherOk)
      {
        // The host version trusts 'bytes'; a kernel that walked a truncated
        // batch would read outside the DMA buffer, so check up front
        if (bytes < 2 * size_t(_lw)) {
          _EB_WARN("batch of %zu bytes is shorter than 2 lines of %u bytes", bytes, _lw);
          _next   = nullptr;
          _status = EvtBatcherTooShort;
        }
      }

      // Iterate backwards over the subframes.  Returns nullptr when the batch
      // is exhausted or found to be corrupt; check status() to tell which.
      __device__
      const EvtBatcherSubFrameTail* next()
      {
        const EvtBatcherSubFrameTail* const save = (const EvtBatcherSubFrameTail*)_next;
        if (!save)  return save;         // no more subframes

        if (save->size() == 0) {         // drp/ logs this as critical
          _EB_WARN("found corrupt size=0");
          _next   = nullptr;
          _status = EvtBatcherZeroSize;
          return nullptr;
        }

        const uint32_t totSize = save->totSize();

        // see if we've jumped backwards too far
        if (_next - totSize < _end) {
          _EB_WARN("corrupt output: %ld %u", (long)(_next - _end), totSize);
          _next   = nullptr;
          _status = EvtBatcherCorrupt;
          return nullptr;
        }

        // compute the next subframe tail ptr
        if (_next - totSize == _end) {
          // indicates this is the last one
          _next = nullptr;
        } else {
          _next -= totSize + _lw;
          // drp/ leaves this for the next iteration to catch, by which point it
          // has already dereferenced memory ahead of the batch
          if (_next < _end) {
            _EB_WARN("corrupt output: tail underruns the batch by %ld bytes", (long)(_end - _next));
            _next   = nullptr;
            _status = EvtBatcherCorrupt;
            return nullptr;
          }
        }
        return save;
      }

      __device__ unsigned status() const { return _status; }
      __device__ bool     ok()     const { return _status == EvtBatcherOk; }
    private:
      unsigned       _lw;
      const uint8_t* _next;
      const uint8_t* _end;
      unsigned       _status;
    };

    // One sub-frame of a scanned batch.  The offset is relative to the start of
    // the payload handed to scan(), which is what lets a scan be reused for
    // every event that lands in a different DMA buffer.
    struct EvtBatcherSubFrame {
      uint32_t offset;
      uint32_t size;

      __device__ const uint8_t* data(const uint8_t* const __restrict__ payload) const
      { return payload + offset; }
    };

    // A cached scan of one batch.
    //
    // Walking the tails is serial, so one thread does it once and every thread
    // then reads the flat, tdest-indexed result.  Correctly operating hardware
    // repeats the same layout on every L1Accept, so the walk is redone only when
    // the payload size differs from the one that produced the cached scan;
    // Reader::_waitForDMA() decides that via matches().
    //
    // A Detector that has the AxiStream Batcher implemented batches its
    // transitions too, not just its L1Accepts -- see Drp::BEBDetector::
    // getTimingHeader(), which steps over a batcher header unconditionally.  A
    // transition's batch simply has no data sub-frames.  Its size therefore
    // differs from an L1Accept's, so the size of the last such batch is latched
    // separately, in noDataBytes; without that, every transition would evict the
    // L1Accept scan and the two would rescan alternately forever.
    //
    // Holes are left where a tdest is absent (offset == 0, size == 0), matching
    // the indexing of Drp::BEBDetector::_subframes().
    //
    // Host-constructed, then copied to the device, following the RingIndexDtoD
    // idiom.  The scan state lives twice: in device global memory, which is what
    // the per-event hot path reads, and in a pinned mirror the host can read
    // without a transfer.  The mirror is written only when a scan happens, so it
    // costs nothing per event.
    class EvtBatcherSubFrames
    {
    public:
      struct Scan {
        size_t   bytes;                 // Payload size that produced this scan; 0 = none yet
        size_t   noDataBytes;           // Size of the last batch bearing no data
        unsigned count;                 // Highest tdest present, plus one
        unsigned status;                // An EvtBatcherStatus
      };
    public:
      // 'firstData' is Gpu::Detector::firstDataSubframe(): a batch whose highest
      // tdest is below it carries no detector data, which is how a transition is
      // told from an L1Accept.
      __host__ EvtBatcherSubFrames(unsigned capacity, unsigned firstData) :
        m_sub      (nullptr),
        m_scan     (nullptr),
        m_mirror   (nullptr),
        m_capacity (capacity),
        m_firstData(firstData)
      {
        chkError(cudaMalloc(&m_sub,    capacity * sizeof(*m_sub)));
        chkError(cudaMemset( m_sub, 0, capacity * sizeof(*m_sub)));
        chkError(cudaMalloc(&m_scan,    sizeof(*m_scan)));
        // Pinned, so both the device and the host can reach it (UVA), as is
        // done for the Reader's state metrics
        chkError(cudaHostAlloc(&m_mirror, sizeof(*m_mirror), cudaHostAllocDefault));
        reset();
      }

      __host__ ~EvtBatcherSubFrames()
      {
        if (m_mirror)  chkError(cudaFreeHost(m_mirror));
        if (m_scan)    chkError(cudaFree(m_scan));
        if (m_sub)     chkError(cudaFree(m_sub));
      }

      // Discard the cached scan, forcing a rescan on the next L1Accept.  Call
      // on (re-)Configure, since the payload layout may have changed.
      __host__ void reset()
      {
        const Scan empty{0, 0, 0, EvtBatcherOk};
        *m_mirror = empty;
        chkError(cudaMemcpy(m_scan, &empty, sizeof(empty), cudaMemcpyDefault));
        chkError(cudaMemset(m_sub, 0, m_capacity * sizeof(*m_sub)));
      }

      // How the last scan fared, read straight out of pinned memory
      __host__ unsigned lastStatus()      const { return m_mirror->status; }
      __host__ unsigned lastCount()       const { return m_mirror->count; }
      __host__ size_t   lastBytes()       const { return m_mirror->bytes; }
      __host__ size_t   lastNoDataBytes() const { return m_mirror->noDataBytes; }
      __host__ bool     lastOk()          const { return m_mirror->status == EvtBatcherOk; }

    public:
      // True when the cached scan describes a payload of this size.  One load
      // from device memory, so cheap enough for the per-event path.
      __device__ bool matches(size_t bytes) const
      { return (bytes == m_scan->bytes) && (m_scan->status == EvtBatcherOk); }

      // True when a batch of this size has already been found to bear no data, so
      // there is nothing to rescan and nothing for a policy to interpret
      __device__ bool matchesNoData(size_t bytes) const
      { return (bytes != 0) && (bytes == m_scan->noDataBytes); }

      __device__ unsigned status()    const { return m_scan->status; }
      __device__ unsigned count()     const { return m_scan->count; }
      __device__ unsigned capacity()  const { return m_capacity; }
      __device__ unsigned firstData() const { return m_firstData; }
      __device__ bool     ok()        const { return m_scan->status == EvtBatcherOk; }

      __device__ const EvtBatcherSubFrame& operator[](unsigned tdest) const
      { return m_sub[tdest]; }

      // Walk the batch and cache the result.  Single-threaded: the caller must
      // ensure exactly one thread calls this.  Returns an EvtBatcherStatus.
      //
      // 'payload' is the DMA payload, i.e. the DMA buffer past the line bearing
      // the DmaDsc, which this code deliberately knows nothing about.  For a
      // Detector that presents sub-frames the batch begins there, so the batcher
      // header is at payload[0] and the TimingHeader is sub-frame 0's data.
      //
      // Walked twice: once to count the sub-frames, and again to record them only
      // if the batch bears data.  That keeps a transition's batch from evicting
      // the L1Accept layout that the data path depends on.  Both passes are a
      // handful of iterations over the tails, and happen only when a size is met
      // for the first time.
      __device__
      unsigned scan(const uint8_t* const __restrict__ payload, size_t bytes)
      {
        // Pass one: how many sub-frames, and is the batch intelligible?
        unsigned count  = 0;
        unsigned status = EvtBatcherOk;
        {
          EvtBatcherIterator            it((const EvtBatcherHeader*)payload, bytes);
          const EvtBatcherSubFrameTail* tail;
          while ((tail = it.next())) {
            const unsigned tdest = tail->tdest();
            if (tdest >= m_capacity) {
              _EB_WARN("tdest %u exceeds the expected sub-frame count %u", tdest, m_capacity);
              status = EvtBatcherOverflow;
              break;
            }
            // Tails are walked highest tdest first, so this settles immediately
            if (tdest + 1 > count)  count = tdest + 1;
          }
          if (status == EvtBatcherOk)  status = it.status();
        }

        if (status != EvtBatcherOk) {   // Unintelligible: latch neither size
          m_scan->count  = count;
          m_scan->status = status;
          *m_mirror      = *m_scan;
          return status;
        }

        if (count <= m_firstData) {     // A transition: no data sub-frames
          m_scan->noDataBytes = bytes;
          m_scan->count       = count;
          m_scan->status      = status;
          *m_mirror           = *m_scan;
          return status;
        }

        // Pass two: record the data layout, which is what a policy reads
        for (unsigned i = 0; i < m_capacity; ++i) {
          m_sub[i].offset = 0;
          m_sub[i].size   = 0;
        }
        EvtBatcherIterator            it((const EvtBatcherHeader*)payload, bytes);
        const EvtBatcherSubFrameTail* tail;
        while ((tail = it.next())) {
          const unsigned tdest = tail->tdest();
          m_sub[tdest].offset = uint32_t(tail->data() - payload);
          m_sub[tdest].size   = tail->size();
        }

        // Cache for reuse, and publish to the host
        m_scan->bytes  = bytes;
        m_scan->count  = count;
        m_scan->status = status;
        *m_mirror      = *m_scan;
        return status;
      }

    private:
      EvtBatcherSubFrame* m_sub;        // [m_capacity], device global memory
      Scan*               m_scan;       // Device global memory: the hot path
      Scan*               m_mirror;     // Pinned: the host's view of the above
      unsigned            m_capacity;   // Gpu::Detector::subframeCount()
      unsigned            m_firstData;  // Gpu::Detector::firstDataSubframe()
    };

  } // Gpu
} // Drp
