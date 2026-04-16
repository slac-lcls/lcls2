#include "TransitionCache.hh"
#include "xtcdata/xtc/Dgram.hh"

using namespace XtcData;
using namespace psalg::shmem;

//#define DBUG
static const int MAX_SLOW_UPDATES = 3;

TransitionCache::TransitionCache(char* p, size_t sz, unsigned nbuff) :
  _pShm(p),
  _szShm(sz),
  _numberofTrBuffers(nbuff),
  _not_ready(0),
  _allocated(new unsigned[nbuff]),
  _nSlowUpdates(0)
{
  sem_init(&_sem, 0, 1);
  memset(_allocated, 0, _numberofTrBuffers*sizeof(unsigned));

  for(unsigned i=0; i<_numberofTrBuffers; i++) {
    Dgram* dg = new (p + _szShm*i) Dgram;
    unsigned env = 0;
    Transition tr(Dgram::Event, TransitionId::Reset,
                  TimeStamp(0,0), env);
    new(dg) Dgram(tr);
    _freeTr.push_back(i);
  }
}

TransitionCache::~TransitionCache()
{
  sem_destroy(&_sem);
  delete[] _allocated;
}

void TransitionCache::dump() const {
  printf("---TransitionCache---\n");
  printf("\tBuffers:\n");
  for(unsigned i=0; i<_numberofTrBuffers; i++) {
    const Dgram& odg = *reinterpret_cast<const Dgram*>(_pShm + _szShm*i);
    time_t t=odg.time.seconds();
    char cbuf[64]; ctime_r(&t,cbuf); strtok(cbuf,"\n");
    printf ("%15.15s : %s : [%d] = %08x\n",
            TransitionId::name(odg.service()),
            cbuf,
            i, _allocated[i]);
  }
  std::stack<int> cached(_cachedTr);
  printf("\tCached: ");
  while(!cached.empty()) {
    printf("%d ",cached.top());
    cached.pop();
  }
  printf("\n\tFree: ");
  for(std::list<int>::const_iterator it=_freeTr.begin();
      it!=_freeTr.end(); it++)
    printf("%d ",*it);
  printf("\n");
}

std::stack<int> TransitionCache::current() {
  sem_wait(&_sem);
  std::stack<int> cached(_cachedTr);
  std::stack<int> tr;
  while(!cached.empty()) {
    tr.push(cached.top());
    cached.pop();
  }
  sem_post(&_sem);
  return tr;
}

//
//  Find a free buffer for the next transition
//
int  TransitionCache::allocate  (TransitionId::Value id) {
  int result = -1;
  TransitionId::Value oid = _cachedTr.empty() ? 
    TransitionId::Reset :
    reinterpret_cast<const Dgram*>(_pShm + _szShm*_cachedTr.top())->service();
#ifdef DBUG
  printf("allocate(%s) [old %s]\n",
         TransitionId::name(id),
         TransitionId::name(oid));
  for(unsigned i=0; i<_numberofTrBuffers; i++)
    printf("%08x ",_allocated[i]);
  printf("\n");
#endif
  bool lbegin = ((id&1)==0);
  sem_wait(&_sem);

  for(std::list<int>::iterator it=_freeTr.begin();
      it!=_freeTr.end(); it++)
    if (_allocated[*it]==0) {

      unsigned ibuffer = *it;

      //
      //  Cache the transition for any clients
      //    which may not be listening (yet)
      //
      if ( _cachedTr.empty() ) {  // First transition
        if (id==TransitionId::Configure) {
          _freeTr.remove(ibuffer);
          _cachedTr.push(ibuffer);
        }
        else {
          fprintf(stderr, "Unexpected state for TransitionCache: _cachedTr empty but tr[%s]!=Configure\n",
                 TransitionId::name(id));
          //dump();
          //abort();
        }
      }
      else {
        //  A limited # of SlowUpdates are cached
        if (id == TransitionId::SlowUpdate) {
#ifdef DBUG
          printf("allocate nSlowUpdates %d\n",_nSlowUpdates);
#endif
          if (_nSlowUpdates < MAX_SLOW_UPDATES) {
            ++_nSlowUpdates;
            //  Don't need to move off the free transition (nothing puts it back)
            // _freeTr.remove(ibuffer);  // Fill a free buffer
            // _cachedTr.push(ibuffer);  // and push it onto the cache
          }
        }
        else if (id == oid+2) {       // Next begin transition
          _freeTr.remove(ibuffer);
          _cachedTr.push(ibuffer);
        }
        else if (id == oid+1) {  // Matching end transition
          int ib=_cachedTr.top();  // Return the matching begin tr
          _cachedTr.pop();         // to the free stack
          _freeTr.push_back(ib);
        }
        else if (id  == TransitionId::Disable) {
          //  Free all the SlowUpdates and the matching Enable
          while (oid > TransitionId::BeginStep) {
            int ib=_cachedTr.top();
            _cachedTr.pop();
            _freeTr.push_back(ib);
            //  Special count of SlowUpdates returned to free stack (no clients)
            if (oid == TransitionId::SlowUpdate && _allocated[ib]==0)
              --_nSlowUpdates;
            oid = reinterpret_cast<const Dgram*>(_pShm + _szShm*_cachedTr.top())->service();
          }
        }
        else {  // unexpected transition
          fprintf(stderr, "Unexpected transition for TransitionCache: tr[%s]!=[%s] or [%s]\n",
                 TransitionId::name(id),
                 TransitionId::name(TransitionId::Value(oid+2)),
                 TransitionId::name(TransitionId::Value(oid+1)));
          if (lbegin) { // Begin transition
            if (id > oid) {  // Missed a begin transition leading up to it
              fprintf(stderr, "Irrecoverable.\n");
              dump();
              abort();
            }
            else {
              fprintf(stderr, "Recover by rolling back.\n");
              do {
                int ib=_cachedTr.top();
                _freeTr.push_back(ib);
                oid = reinterpret_cast<const Dgram*>(_pShm + _szShm*ib)->service();
                _cachedTr.pop();
              } while(oid > id);
              _freeTr.remove(ibuffer);
              _cachedTr.push(ibuffer);
            }
          }
          else { // End transition
            fprintf(stderr, "Recover by rolling back.\n");
            while( id < oid+3 ) {
              int ib=_cachedTr.top();
              _freeTr.push_back(ib);
              _cachedTr.pop();
              if (_cachedTr.empty()) break;
              oid = reinterpret_cast<const Dgram*>(_pShm + _szShm*_cachedTr.top())
                ->service();
            }
          }
        }
      }

      if (lbegin) {
        unsigned not_ready=0;
        for(unsigned itr=0; itr<_numberofTrBuffers; itr++) {
          if (itr==ibuffer) continue;
          const Dgram& odg = *reinterpret_cast<const Dgram*>(_pShm + _szShm*itr);
          if (odg.service()==TransitionId::Enable)
            not_ready |= _allocated[itr];
        }

        // Ignore not_ready on SlowUpdate
        if (id != TransitionId::SlowUpdate && not_ready &~_not_ready)
          printf("Transition %s: not_ready %x -> %x\n",
                 TransitionId::name(id), _not_ready, _not_ready|not_ready);

        _not_ready |= not_ready;
      }

#ifdef DBUG
      printf("not_ready %08x\n",_not_ready);
#endif
      result = ibuffer;
      break;
    }

  sem_post(&_sem);

  return result;
}

//
//  Queue this transition for a client
//
bool TransitionCache::allocate  (int ibuffer, unsigned client) {

  bool result = true;
#ifdef DBUG
  printf("allocate[%d,%d] not_ready %08x\n",ibuffer,client,_not_ready);
#endif

  sem_wait(&_sem);

  if (_not_ready & (1<<client)) {
    TransitionId::Value last=TransitionId::NumberOf;
    for(unsigned i=0; i<_numberofTrBuffers; i++)
      if (_allocated[i] & (1<<client)) {
        TransitionId::Value td =
          reinterpret_cast<const Dgram*>(_pShm + _szShm*i)->service();
        if ((td&1)==1 && td<last) last=td;
      }

    // Ignore _not_ready on SlowUpdate
    TransitionId::Value id =
      reinterpret_cast<const Dgram*>(_pShm + _szShm*ibuffer)->service();
    if (id != TransitionId::SlowUpdate && !((id&1)==1 && id<last))
      result=false;
  }

  if (result)
    _allocated[ibuffer] |= (1<<client);

  sem_post(&_sem);

#ifdef DBUG
  printf("_allocated[%d] = %08x\n",ibuffer,_allocated[ibuffer]);
#endif
  return result;
}

//
//  Client has completed this transition.
//  Remove client from _allocated list for this buffer.
//  Return true if client was previously "not ready" but now is "ready"
bool TransitionCache::deallocate(int ibuffer, unsigned client) {
  bool result=false;
  sem_wait(&_sem);
  { unsigned v = _allocated[ibuffer] & ~(1<<client);
#ifdef DBUG
    printf("_deallocate[%d,%d] %08x -> %08x\n",ibuffer,client,
           _allocated[ibuffer],v);
#else
    if ( _allocated[ibuffer]==v )
      printf("_deallocate[%d,%d] %08x no change\n",ibuffer,client,v);
#endif
    _allocated[ibuffer]=v; 
    if (v==0) {
      const Dgram& odg = *reinterpret_cast<const Dgram*>(_pShm + _szShm*ibuffer);
      TransitionId::Value oid = odg.service();
      if (oid == TransitionId::SlowUpdate) {
        --_nSlowUpdates;
#ifdef DBUG
        fprintf(stderr,"deallocate nSlowUpdates %d\n",_nSlowUpdates);
#endif
      }
    }
  }
  if (_not_ready & (1<<client)) {
    for(unsigned i=0; i<_numberofTrBuffers; i++)
      if (_allocated[i] & (1<<client)) {
        sem_post(&_sem);
        return false;
      }
    printf("not_ready %x -> %x\n", _not_ready, _not_ready&~(1<<client));
    _not_ready &= ~(1<<client);
    result=true;
  }
  sem_post(&_sem);
  return result;
}

//
//  Retire this client.
//  Remove the client from the _allocated list for all buffers.
//
void TransitionCache::deallocate(unsigned client) {
  sem_wait(&_sem);
  _not_ready &= ~(1<<client);
  for(unsigned itr=0; itr<_numberofTrBuffers; itr++)
    _allocated[itr] &= ~(1<<client);
  sem_post(&_sem);
#ifdef DBUG
  printf("deallocate %d\n",client);
  for(unsigned i=0; i<_numberofTrBuffers; i++)
    printf("%08x ",_allocated[i]);
  printf("\n");
#endif
}
