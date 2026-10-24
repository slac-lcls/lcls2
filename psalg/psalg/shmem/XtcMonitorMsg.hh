/**
 * @file
 * @brief Declares psalg::shmem::XtcMonitorMsg, a 16-byte record describing shared-memory buffers, with helpers that format shared-memory and queue names.
 */
#ifndef PsAlg_ShMem_XtcMonitorMsg_hh
#define PsAlg_ShMem_XtcMonitorMsg_hh

#include <stdint.h>
#include <stddef.h>

namespace psalg {
  /** Shared-memory monitoring classes (XtcMonitorMsg, XtcMonitorServer, ShmemClient, ...). */
  namespace shmem {
    /**
     * Record of four 32-bit words: a buffer index; a packed word with the number of buffers (bits 7:0), number of queues (bits 15:8) and return queue (bits 23:16); a size word (bits 27:0 used); and a reserved word.
     * Static helpers write the names of the shared memory and message queues for a tag.
     */
    class XtcMonitorMsg {
      enum { SizeMask   = 0x0fffffff };
      enum { SerialShift = 28 };
    public:
      /** Set all fields to 0. */
      XtcMonitorMsg() : _bufferIndex(0),
                        _numberOfBuffers(0),
                        _sizeOfBuffers(0),
                        _reserved(0) {}
      /** Set the buffer index to bufferIndex and all other fields to 0. */
      XtcMonitorMsg(int bufferIndex) : _bufferIndex(bufferIndex),
                                       _numberOfBuffers(0),
                                       _sizeOfBuffers(0),
                                       _reserved(0) {}
      /** Destructor; does nothing. */
      ~XtcMonitorMsg() {};
    public:
      /** Return the buffer index. */
      int bufferIndex     () const { return _bufferIndex; }
      /** Return bits 7:0 of the packed word. */
      int numberOfBuffers () const { return _numberOfBuffers&0xff; }
      /** Return bits 15:8 of the packed word. */
      int numberOfQueues  () const { return (_numberOfBuffers>>8)&0xff; }
      /** Return the size word masked with SizeMask (bits 27:0). */
      size_t sizeOfBuffers() const { return (size_t)_sizeOfBuffers&SizeMask; }
      /** Return true if return_queue() is 0. */
      bool serial         () const { return return_queue()==0; }
      /** Return bits 23:16 of the packed word. */
      int return_queue    () const { return (_numberOfBuffers>>16)&0xff; }
    public:
      /** Set the buffer index to b and return this. */
      XtcMonitorMsg* bufferIndex(int b) {_bufferIndex=b; return this;}
      /** Set bits 7:0 of the packed word to n & 0xff. */
      void numberOfBuffers      (int n) {_numberOfBuffers &= ~0xff; _numberOfBuffers |= ((n&0xff)<<0); }
      /** Set bits 15:8 of the packed word to n & 0xff. */
      void numberOfQueues       (int n) {_numberOfBuffers &= ~0xff00; _numberOfBuffers |= ((n&0xff)<<8); }
      /** Set bits 27:0 of the size word to s & SizeMask, keeping bits 31:28. */
      void sizeOfBuffers        (int s) {_sizeOfBuffers = (_sizeOfBuffers&~SizeMask) | (s&SizeMask);}
      /** Set bits 23:16 of the packed word to q & 0xff. */
      void return_queue         (int q) {_numberOfBuffers &= ~0xff0000; _numberOfBuffers |= ((q&0xff)<<16); }
    public:
      /** Write "/PdsMonitorSharedMemory_" + tag into buffer with sprintf (no length check). */
      static void sharedMemoryName     (const char* tag, char* buffer);
      /** Write "/PdsToMonitorEvQueue_" + tag + "_" + client into buffer with sprintf (no length check). */
      static void eventInputQueue      (const char* tag, unsigned client, char* buffer);
      /** Write "/PdsToMonitorEvQueue_" + tag + "_" + (client+1) into buffer with sprintf, i.e. the eventInputQueue() name of client+1. */
      static void eventOutputQueue     (const char* tag, unsigned client, char* buffer);
      /** Write "/PdsToMonitorTrQueue_" + tag + "_" + client into buffer with sprintf (no length check). */
      static void transitionInputQueue (const char* tag, unsigned client, char* buffer);
      /** Write "/PdsFromMonitorDiscovery_" + tag into buffer with sprintf (no length check). */
      static void discoveryQueue       (const char* tag, char* buffer);
      /** Write "/PdsToMonitorDiscovery_" + tag + "_" + id into buffer with sprintf (no length check). */
      static void registerQueue        (const char* tag, char* buffer, int id);
    private:
      int32_t  _bufferIndex;
      int32_t  _numberOfBuffers;
      uint32_t _sizeOfBuffers; // hoping we don't get larger than 4GB and SizeMask matters
      uint32_t _reserved;
    };
  };
};

#endif
