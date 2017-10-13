/*
** ++
**  Package:
**
**  Abstract:
**
**
**  Author:
**      Michael Huffer, SLAC, (415) 926-4269
**
**  Creation Date:
**	000 - June 1,1998
**
**  Revision History:
**	None.
**
** --
*/

#ifndef Eb_EbEpoch_hh
#define Eb_EbEpoch_hh

#include <cstddef>

#include "EbEvent.hh"

#include "psdaq/service/LinkedList.hh"

namespace Pds {
  namespace Eb {

#define EbEventList LinkedList<EbEvent> // Notational convenience...

    class EbEpoch : public pds::LinkedList<EbEpoch>
    {
    public:
      void* operator new(size_t, Heap*); // Revisit: Heap
      void  operator delete(void* buffer);
    public:
      EbEpoch(uint64_t key, EbEpoch* after);
      ~EbEpoch();
    public:
      void dump(int number);
    public:
      EbEventList pending;              // Listhead, events pending;
      uint64_t    key;                  // Epoch sequence number
    };
  };
};

/*
** ++
**
**    To speed up the allocation/deallocation of events, they have their
**    own specific "new" and "delete" operators, which work out of a heap
**    of a fixed number of fixed size buffers (the size is set to the size
**    of this object). The heap is established by "Eb" and events are
**    allocated/deallocated by "EbServer".
**
** --
*/

inline void* Pds::Eb::EbEpoch::operator new(size_t size, Heap* heap)
{
  return heap->alloc();
}

/*
** ++
**
**    To speed up the allocation/deallocation of events, they have their
**    own specific "new" and "delete" operators, which work out of a heap
**    of a fixed number of fixed size buffers (the size is set to the size
**    of this object). The heap is established by "EventBuilder" and events
**    are allocated/deallocated by "EbServer".
**
** --
*/

inline void Pds::Eb::EbEpoch::operator delete(void* buffer)
{
  Heap::free(buffer);
}

/*
** ++
**
**    As soon as an event becomes "complete" its datagram is the only
**    information of value within the event. Therefore, when the event
**    becomes complete it is deleted which cause the destructor to
**    remove the event from the pending queue.
**
** --
*/

inline Pds::Eb::EbEpoch::EbEpoch(uint64_t key, EbEpoch* after) :
  pending(),
  key(key)
{
  connect(after);
}

/*
** ++
**
**    As soon as an event becomes "complete" its datagram is the only
**    information of value within the event. Therefore, when the event
**    becomes complete it is deleted which cause the destructor to
**    remove the event from the pending queue.
**
** --
*/

inline Pds::Eb::EbEpoch::~EbEpoch()
{
  disconnect();
}

#endif
