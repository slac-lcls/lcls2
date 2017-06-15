#ifndef PDS_TIMESTAMP_HH
#define PDS_TIMESTAMP_HH

#include <stdint.h>

namespace Pds {
  class TimeStamp {
  public:
    enum {NumFiducialBits = 17};
    enum {MaxFiducials = (1<<17)-32};
    enum {ErrFiducial = (1<<17)-1};
  public:
    TimeStamp();
    TimeStamp(const TimeStamp&);
    TimeStamp(const TimeStamp&, unsigned control);
    TimeStamp(unsigned ticks, unsigned fiducials, unsigned vector, unsigned control=0);

  public:
    unsigned ticks    () const;  // 119MHz counter within the fiducial for
                                 //   eventcode which initiated the readout
    unsigned fiducials() const;  // 360Hz pulse ID
    unsigned control  () const;  // internal bits for alternate interpretation
                                 //   of XTC header fields
    unsigned vector   () const;  // 15-bit seed for event-level distribution
                                 //   ( events since configure )
  public:
    TimeStamp& operator= (const TimeStamp&);
    bool       operator==(const TimeStamp&) const;
    bool       operator>=(const TimeStamp&) const;
    bool       operator<=(const TimeStamp&) const;
    bool       operator< (const TimeStamp&) const;
    bool       operator> (const TimeStamp&) const;

  private:
    uint32_t _low;
    uint32_t _high;
  };
}

#endif
