/**
 * @file
 * @brief EbEvent, one event being built: the contributions received so far for a pulse ID and the contributors still missing.
 */
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

    /** One event being built by EventBuilder: its contributions (a variable-length array that follows the object), the bit list of contributors still missing, accumulated damage and payload size. Allocated from a pool sized by EventBuilder. */
    class EbEvent : public LinkedList<EbEvent>
    {
    private:
      using time_point_t = std::chrono::time_point<fast_monotonic_clock>;
    public:
      /** Macro from Pool.hh that declares an operator new allocating from a Pds::Pool and an operator delete that returns the buffer to its pool. */
      PoolDeclare;
    public:
      /** Start an event with the first contribution ctrb, remove its source bit from contract to form the remaining list, and insert the event into the list after after. Throws a C string if the source is not in contract. */
      EbEvent(uint64_t            contract,
              EbEvent*            after,
              const Pds::EbDgram* ctrb,
              unsigned            immData,
              const time_point_t& t0);
      /** Does nothing; empty body. */
      ~EbEvent();
    public:
      /** Return the immediate data kept for this event (from the first contribution, replaced by a later contribution whose value is greater than MAX_ENTRIES). */
      unsigned        immData()   const;
      /** Return the pulse ID of the first contribution. */
      uint64_t        sequence()  const;
      /** Return the sum of the XTC payload sizes of all contributions, in bytes. */
      size_t          size()      const;
      /** Return the bit list of contributors that have not yet contributed; 0 means complete. */
      uint64_t        remaining() const;
      /** Return the bit list of contributors expected for this event. */
      uint64_t        contract()  const;
      /** Return the accumulated damage. */
      XtcData::Damage damage()    const;
      /** Increase the accumulated damage by value (XtcData::Damage::increase()). */
      void            damage(XtcData::Damage::Value);
    public:
      /** Return the first contribution, which created the event. */
      const Pds::EbDgram*  const  creator() const;
      /** Return a pointer to the first element of the contributions array. */
      const Pds::EbDgram*  const* begin()   const;
      /** Return a pointer one past the last contribution. */
      const Pds::EbDgram** const  end()     const;
    public:
      /** Print one line about the event (pulse ID, control, env, size, source, remaining and contract bits, age and latency) to stderr; with non-zero detail, also number and the list pointers. */
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
