/**
 * @file
 * @brief EbEpoch, an element of the event builder's list of epochs, each holding its pending events.
 */
#ifndef Eb_EbEpoch_hh
#define Eb_EbEpoch_hh

#include <cstddef>

#include "EbEvent.hh"

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Pool.hh"

namespace Pds {
  namespace Eb {

    /** List element for one epoch of the event builder; holds the list of events pending in that epoch. */
    class EbEpoch : public Pds::LinkedList<EbEpoch>
    {
    public:
      /** Macro from Pool.hh that declares an operator new allocating from a Pds::Pool and an operator delete that returns the buffer to its pool. */
      PoolDeclare;
    public:
      /** Construct with an empty pending list and the given key, and insert this epoch into the list after after. */
      EbEpoch(uint64_t key, EbEpoch* after);
      /** Remove this epoch from its list. */
      ~EbEpoch();
    public:
      /** Print this epoch (with its list pointers when detail is non-zero) and its pending events (all of them when detail is non-zero, otherwise only the first) to stderr; number labels the epoch. */
      void dump(unsigned detail, int number);
    public:
      /** List head of the events pending in this epoch. */
      LinkedList<EbEvent> pending;    // Listhead, events pending;
      /** Epoch sequence number (key). */
      uint64_t            key;        // Epoch sequence number
    };
  };
};

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
