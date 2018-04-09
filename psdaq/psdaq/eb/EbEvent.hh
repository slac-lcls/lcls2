#ifndef Eb_EbEvent_hh
#define Eb_EbEvent_hh

#include <stdint.h>

#include "EbContribution.hh"

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Pool.hh"


namespace XtcData {
  class Dgram;
};

namespace Pds {

  class GenericPool;

  namespace Eb {

    class EventBuilder;

    class EbEvent : public LinkedList<EbEvent>
    {
    public:
      PoolDeclare;
    public:
      EbEvent(uint64_t              contract,
              EventBuilder*         eb,
              EbEvent*              after,
              const XtcData::Dgram* contrib,
              uint64_t              mask);
      virtual ~EbEvent();
    public:
      uint64_t sequence()  const;
      size_t   size()      const;
      uint64_t remaining() const;
      uint64_t contract()  const;
    public:
      const EbContribution*  const  creator() const;
      const EbContribution*  const* begin()   const;
      const EbContribution** const  end()     const;
    public:
      void     dump(int number);
    private:
      friend class EventBuilder;
    private:
      EbEvent* _add(const XtcData::Dgram*);
      void     _insert(const XtcData::Dgram*);
      bool     _alive();
    private:
      uint64_t               _sequence;        // Event's sequence identifier
      size_t                 _size;            // Total contribution size (in bytes)
      uint64_t               _remaining;       // List of clients which have contributed
      uint64_t               _contract;        // -> potential list of contributors
      EventBuilder*          _eb;              // -> Back-end processing object
      int                    _living;          // Aging counter
      uint64_t               _key;             // Masked epoch
      const EbContribution** _last;            // Pointer into the contributions array
      const EbContribution*  _contributions[]; // Array of contributions
    };
  };
};

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

inline const Pds::Eb::EbContribution* const Pds::Eb::EbEvent::creator() const
{
  return _contributions[0];
}

inline const Pds::Eb::EbContribution* const* Pds::Eb::EbEvent::begin() const
{
  return _contributions;
}

inline const Pds::Eb::EbContribution** const Pds::Eb::EbEvent::end() const
{
  return _last;
}

/*
** ++
**
**   In principle an event could sit on the pending queue forever,
**   waiting for its contract to complete. To handle this scenario,
**   the EB times-out the oldest event on the pending queue.
**   This is accomplished by periodically calling this function,
**   which counts down a counter whose initial value is specified
**   at construction time. When the counter drops to less than
**   or equal to zero the event is expired. Note: the arrival of
**   any contributor to this event will RESET the counter.
**
** --
*/

inline bool Pds::Eb::EbEvent::_alive()
{
  return --_living > 0;
}

#endif
