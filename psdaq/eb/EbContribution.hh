#ifndef Pds_Eb_EbContribution_hh
#define Pds_Eb_EbContribution_hh

#include <stdint.h>

#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/LinkedList.hh"
#include "psdaq/service/Pool.hh"


namespace Pds {
  namespace Eb {

#define EbCntrbList LinkedList<EbContribution>

    class EbContribution : public EbCntrbList
    {
    public:
      PoolDeclare;
    public:
      EbContribution(const XtcData::Dgram* datagram, uint64_t appParm);
      ~EbContribution();
    public:
      unsigned  payloadSize()   const;
      unsigned  number()        const;
      uint64_t  numberAsMask()  const;
      uint64_t  retire()        const;
      uint64_t  data()          const;
    public:
      const XtcData::Dgram* datagram() const;
    private:
      const XtcData::Dgram* _datagram;
      uint64_t              _appParam;
    };
  };
};

/*
** ++
**
**   Return the size (in bytes) of the contribution's payload
**
** --
*/

inline unsigned Pds::Eb::EbContribution::payloadSize() const
{
  return _datagram->xtc.sizeofPayload();
}

/*
** ++
**
**   Return the size (in bytes) of the contribution's payload
**
** --
*/

inline Pds::Eb::EbContribution::EbContribution(const XtcData::Dgram* datagram,
                                               uint64_t              appParam) :
  _datagram(datagram),
  _appParam(appParam)
{
}

/*
** ++
**
**   Return the size (in bytes) of the contribution's payload
**
** --
*/

inline Pds::Eb::EbContribution::~EbContribution()
{
  disconnect();
}

/*
** ++
**
**    Return the source ID of the contributor which sent this packet...
**
** --
*/

inline unsigned Pds::Eb::EbContribution::number() const
{
  return _datagram->xtc.src.log() & 0x00ffffff; // Revisit: Shouldn't need to mask here
}

/*
** ++
**
**   Return the source ID of the contributor which sent this packet as a mask...
**
** --
*/

inline uint64_t Pds::Eb::EbContribution::numberAsMask() const
{
  return 1ull << number();
}

/*
** ++
**
**   Return the (complemented) value of the source ID.  The value is used by
**   "EbEvent" to strike down its "remaining" field which in turn signifies its
**   remaining contributors.
**
** --
*/

inline uint64_t Pds::Eb::EbContribution::retire() const
{
  return ~numberAsMask();
}

/*
** ++
**
**   Provide access to the application free parameter.
**
** --
*/

inline uint64_t Pds::Eb::EbContribution::data() const
{
  return _appParam;
}

/*
** ++
**
**   Provide access to the source datagram.
**
** --
*/

inline const XtcData::Dgram* Pds::Eb::EbContribution::datagram() const
{
  return _datagram;
}

#endif
