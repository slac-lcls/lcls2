#ifndef Eb_EbEvent_hh
#define Eb_EbEvent_hh

#include <stdint.h>

#include "eb.hh"

#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Pool.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/fast_monotonic_clock.hh"


namespace Pds {

  class GenericPool;

  namespace Eb {

    class EventBuilder;

    class EbEvent : public LinkedList<EbEvent>
    {
    private:
      using time_point_t = std::chrono::time_point<fast_monotonic_clock>;
    public:
      PoolDeclare;
    public:
      EbEvent(uint64_t            contract,
              EbEvent*            after,
              const Pds::EbDgram* ctrb,
              unsigned            immData,
              const time_point_t& t0);
      ~EbEvent();
    public:
      unsigned        immData()   const;
      uint64_t        sequence()  const;
      size_t          size()      const;
      uint64_t        remaining() const;
      uint64_t        contract()  const;
      XtcData::Damage damage()    const;
      void            damage(XtcData::Damage::Value);
    public:
      const Pds::EbDgram*  const  creator() const;
      const Pds::EbDgram*  const* begin()   const;
      const Pds::EbDgram** const  end()     const;
    public:
      void     dump(unsigned detail, int number);
    private:
      friend class EventBuilder;
    private:
      EbEvent* _add(const Pds::EbDgram*, unsigned immData);
      void     _insert(const Pds::EbDgram*);
    private:
      size_t               _size;            // Total contribution size (in bytes)
      uint64_t             _remaining;       // List of clients which have contributed
      const uint64_t       _contract;        // -> potential list of contributors
      time_point_t         _t0;              // Starting time of timeout
      unsigned             _immData;         // A contribution's immediate data
      XtcData::Damage      _damage;          // Accumulate damage about this event
      const Pds::EbDgram** _last;            // Pointer into the contributions array
      const Pds::EbDgram*  _contributions[]; // Array of contributions
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
**    This function is used to insert a "dummy" contribution into the event.
**    The dummy contribution is identified by the input argument.
**
** --
*/

inline void Pds::Eb::EbEvent::_insert(const EbDgram* dummy)
{
  *_last++ = dummy;
}

/*
** ++
**
**    Give EventBuilder user interface access to the immediate data.
**
** --
*/

inline unsigned Pds::Eb::EbEvent::immData() const
{
  return _immData;
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
  return creator()->pulseId();
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
**   Method to retrieve the event's damage value.
**
** --
*/

inline XtcData::Damage Pds::Eb::EbEvent::damage() const
{
  return _damage;
}

/*
** ++
**
**   Method to increase the event's damage value.
**
** --
*/

inline void Pds::Eb::EbEvent::damage(XtcData::Damage::Value value)
{
  _damage.increase(value);
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

inline const Pds::EbDgram* const Pds::Eb::EbEvent::creator() const
{
  return _contributions[0];
}

inline const Pds::EbDgram* const* Pds::Eb::EbEvent::begin() const
{
  return _contributions;
}

inline const Pds::EbDgram** const Pds::Eb::EbEvent::end() const
{
  return _last;
}

#endif
