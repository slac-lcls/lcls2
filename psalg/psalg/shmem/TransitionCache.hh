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
    class TransitionCache {
    public:
      TransitionCache(char* p, size_t sz, unsigned);
      ~TransitionCache();
    public:
      void dump() const;
      std::stack<int> current();
      int  allocate  (XtcData::TransitionId::Value);
      bool allocate  (int ibuffer, unsigned client);
      bool deallocate(int ibuffer, unsigned client);
      void deallocate(unsigned client);
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
    };
  };
};

#define TR_HASH(a,b)                (((a)*1000)+(b))

#define CONFIGURE_UNCONFIGURE       TR_HASH(XtcData::TransitionId::Configure, XtcData::TransitionId::Unconfigure)
#define CONFIGURE_BEGINRUN          TR_HASH(XtcData::TransitionId::Configure, XtcData::TransitionId::BeginRun)

#define ENABLE_DISABLE              TR_HASH(XtcData::TransitionId::Enable, XtcData::TransitionId::Disable)
#define ENABLE_SLOWUPDATE           TR_HASH(XtcData::TransitionId::Enable, XtcData::TransitionId::SlowUpdate)

#define SLOWUPDATE_SLOWUPDATE       TR_HASH(XtcData::TransitionId::SlowUpdate, XtcData::TransitionId::SlowUpdate)
#define SLOWUPDATE_DISABLE          TR_HASH(XtcData::TransitionId::SlowUpdate, XtcData::TransitionId::Disable)

#endif
