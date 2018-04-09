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
#include "psdaq/service/Pool.hh"

namespace Pds {
  namespace Eb {

    class EbEpoch : public Pds::LinkedList<EbEpoch>
    {
    public:
      PoolDeclare;
    public:
      EbEpoch(uint64_t key, EbEpoch* after);
      ~EbEpoch();
    public:
      void dump(int number);
    public:
      LinkedList<EbEvent> pending;      // Listhead, events pending;
      uint64_t            key;          // Epoch sequence number
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
