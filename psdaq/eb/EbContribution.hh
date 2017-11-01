#ifndef Pds_Eb_EbContribution_hh
#define Pds_Eb_EbContribution_hh

#include "xtcdata/xtc/Dgram.hh"

namespace Pds {
  namespace Eb {

    class EbContribution : public XtcData::Dgram
    {
    public:
      unsigned  payloadSize()   const;
      unsigned  number()        const;
      uint64_t  numberAsMask()  const;
      uint64_t  retire()        const;
    public:
      const XtcData::Dgram* datagram() const;
      XtcData::Dgram*       datagram();
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
  return xtc.sizeofPayload();
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
  return xtc.src.log() & 0x00ffffff;    // Revisit: Shouldn't need to mask here
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
**   Provide access to the source datagram.
**
** --
*/

inline const XtcData::Dgram* Pds::Eb::EbContribution::datagram() const
{
  return this;
}

inline XtcData::Dgram* Pds::Eb::EbContribution::datagram()
{
  return this;
}

#endif
