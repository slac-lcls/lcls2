/**
 * @file
 * @brief Declares psalg::shmem::TransitionCache, which tracks transition buffers in shared memory and the clients using them.
 */
#ifndef PsAlg_ShMem_TransitionCache_hh
#define PsAlg_ShMem_TransitionCache_hh

//
//  TransitionCache class - the purpose of this class is to cache transitions for
//    clients and track the transitions which are still to be served to latent clients
//

#include "xtcdata/xtc/TransitionId.hh"

#include <list>
#include <stack>

#include <semaphore.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

namespace psalg {
  namespace shmem {
    /**
     * Tracks the transition buffers in shared memory: a stack of buffers holding the cached transitions of the current state, a list of free buffers, and a per-buffer bitmask of clients still using each buffer.
     * Per the header comment, it caches transitions for clients and tracks those still to be served to late clients. A process-local semaphore guards the state.
     */
    class TransitionCache {
    public:
      /** Use the third argument as the number of buffers, each sz bytes, starting at p; write a Reset transition Dgram into each and mark all of them free with no clients. */
      TransitionCache(char* p, size_t sz, unsigned);
      /** Destroy the semaphore and free the per-buffer client masks. */
      ~TransitionCache();
    public:
      /** Print each buffer's transition name, time and client mask, then the cached and free buffer indexes, to stdout. */
      void dump() const;
      /** Return the cached buffer indexes as a stack whose top is the oldest cached transition (the reverse of the internal order). */
      std::stack<int> current();
      /**
       * Choose the first free buffer that no client holds for transition id and update the cache: Configure starts it, the next begin transition (top id + 2) is pushed, the matching end (top id + 1) pops the top, Disable pops entries above BeginStep, and up to three SlowUpdates are counted.
       * Other transitions are reported on stderr and rolled back, or abort() is called if a begin transition was skipped. For begin (even) ids, clients still holding an Enable buffer are added to the not-ready mask.
       * @return The buffer index, or -1 if no buffer is free.
       */
      int  allocate  (XtcData::TransitionId::Value);
      /**
       * Add client to the client mask of buffer ibuffer and return true.
       * If client is marked not ready, this is refused (returns false) unless the buffer holds a SlowUpdate or an end transition (odd id) lower than every end transition the client already holds.
       */
      bool allocate  (int ibuffer, unsigned client);
      /**
       * Remove client from the mask of buffer ibuffer (printing a message if it was not set); when a SlowUpdate buffer becomes unused the SlowUpdate count is decremented.
       * @return true if client was marked not ready and now holds no buffer, in which case its not-ready bit is cleared; otherwise false.
       */
      bool deallocate(int ibuffer, unsigned client);
      /** Retire client: clear its bit in the not-ready mask and in every buffer's client mask. */
      void deallocate(unsigned client);
      /** Return the bitmask of clients marked not ready (per the member comment, clients that are behind in processing). */
      unsigned not_ready() const { return _not_ready; }
    private:
      sem_t            _sem;
      const char*      _pShm;
      size_t           _szShm;
      unsigned         _numberofTrBuffers;
      unsigned         _not_ready; // bitmask of clients that are behind in processing
      unsigned*        _allocated; // bitmask of clients that are processing
      std::stack <int> _cachedTr;  // set of transitions for the current DAQ state
      std::list  <int> _freeTr;    // complement of _cachedTr
      int              _nSlowUpdates;
    };
  };
};

#endif
