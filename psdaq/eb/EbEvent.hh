#ifndef Eb_EbEvent_hh
#define Eb_EbEvent_hh

#include <stdint.h>

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Pool.hh"


namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    class EventBuilder;
    class EbContribution;

    class EbEvent : public LinkedList<EbEvent>
    {
    public:
      PoolDeclare;
    public:
      EbEvent(uint64_t        contract,
              EventBuilder*   builder,
              EbEvent*        after,
              EbContribution* contrib,
              uint64_t        mask);
      ~EbEvent();
    public:
      uint64_t          sequence()  const;
      size_t            size()      const;
      uint64_t          remaining() const;
      uint64_t          contract()  const;
      void*             data();
    public:
      EbContribution*   creator();
      EbContribution**  start();
      EbContribution**  end();
    public:
      void dump(int number);
    private:
      friend class EventBuilder;
    private:
      EbEvent*         _add(EbContribution*);
      void             _insert(EbContribution*);
      int              _alive();
    private:
      uint64_t         _sequence;     // Event's sequence identifier
      size_t           _size;         // Total contribution size (in bytes)
      uint64_t         _remaining;    // List of clients which have contributed
      uint64_t         _contract;     // -> potential list of contributors
      EbContribution** _head;         // Next    pending contribution
      EbContribution** _tail;         // Base of pending contributions
      EventBuilder*    _eb;           // -> Back-end processing object
      int              _living;       // Aging counter
      uint64_t         _key;          // Masked epoch
    private:
#define NSRC  64
      EbContribution*  _pending[NSRC];  // Pending contribution list
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

inline Pds::Eb::EbEvent::~EbEvent()
{
}

/*
** ++
**
**    Give EventBuilder user interface access to the sequence number.
**
** --
*/

inline uint64_t Pds::Eb::EbEvent::sequence() const
{
  return _sequence;
}

/*
** ++
**
**   Return the size (in bytes) of the event's payload
**
** --
*/

inline size_t Pds::Eb::EbEvent::size() const
{
  return _size;
}

/*
** ++
**
**   Returns a bit-list which specifies the slots expected to contribute
**   to this event. If a bit is SET at a particular offset, the slot
**   corresponding to that offset is an expected contributor.
**
** --
*/

inline uint64_t Pds::Eb::EbEvent::contract() const
{
  return _contract;
}

/*
** ++
**
**   Returns a bit-list which specifies the slots remaining to contribute
**   to this event. If a bit is SET at a particular offset, the slot
**   corresponding to that offset is remaining as a contributor. Consequently,
**   a "complete" event will return a value of zero (0).
**
** --
*/

inline uint64_t Pds::Eb::EbEvent::remaining() const
{
  return _remaining;
}

/*
** ++
**
**    An event comes into existence with the arrival of its first
**    expected contributor. This function will a pointer to the packet
**    corresponding to its first contributor.
**
** --
*/

inline Pds::Eb::EbContribution* Pds::Eb::EbEvent::creator()
{
  return *_tail;
}

inline Pds::Eb::EbContribution** Pds::Eb::EbEvent::start()
{
  return _tail;
}

inline Pds::Eb::EbContribution** Pds::Eb::EbEvent::end()
{
  return _head;
}

/*
** ++
**
**   In principle an event could set on the pending queue forever,
**   waiting for its contract to complete. To handle this scenario,
**   "odfVeb" times-out the oldest event on the pending queue.
**   This is accomplished by periodically calling this function,
**   which counts down a counter whose initial value is specified
**   at construction time. When the counter drops to less than
**   or equal to zero the event is expired. Note: the arrival of
**   any contributor will RESET the counter.
**
** --
*/

inline int Pds::Eb::EbEvent::_alive()
{
  return --_living;
}

#endif
