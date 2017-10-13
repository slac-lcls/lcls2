/*
** ++
**
**  Facility:
**	Service
**
**  Abstract:
**
**  Author:
**      R. Claus, SLAC/PEP-II BaBar Online Dataflow Group
**
**  History:
**	December 10, 1999 - Created
**
**  Copyright:
**                                Copyright 1999
**                                      by
**                         The Board of Trustees of the
**                       Leland Stanford Junior University.
**                              All rights reserved.
**
**
**         Work supported by the U.S. Department of Energy under contract
**       DE-AC03-76SF00515.
**
**                               Disclaimer Notice
**
**        The items furnished herewith were developed under the sponsorship
**   of the U.S. Government.  Neither the U.S., nor the U.S. D.O.E., nor the
**   Leland Stanford Junior University, nor their employees, makes any war-
**   ranty, express or implied, or assumes any liability or responsibility
**   for accuracy, completeness or usefulness of any information, apparatus,
**   product or process disclosed, or represents that its use will not in-
**   fringe privately-owned rights.  Mention of any product, its manufactur-
**   er, or suppliers shall not, nor is it intended to, imply approval, dis-
**   approval, or fitness for any particular use.  The U.S. and the Univer-
**   sity at all times retain the right to use and disseminate the furnished
**   items for any purpose whatsoever.                       Notice 91 02 01
**
** --
*/

#ifndef PDS_SYSCLK_HH
#define PDS_SYSCLK_HH

#ifndef VXWORKS
#include <time.h>
#endif

namespace Pds {
class SysClk
{
public:
  static unsigned sample();
  static double   nsPerTick();
  static unsigned since (unsigned prev);
  static long long int diff(const timespec& end, 
			    const timespec& start);
};
}

#ifdef VXWORKS
/*
** ++
**
**    Return the number of nano-seconds per tick
**
** --
*/

inline double Pds::SysClk::nsPerTick()
{
  return 60.0;                          // 60 nS / tick
}

/*
** ++
**
**    Samples the lower order longword of the processor's time clock
**    and returns that value to the caller. The value returned is in
**    60 nano-second tics...
**
** --
*/

inline unsigned Pds::SysClk::sample()
{
  unsigned  time;
  asm volatile ("mftb %0": "=r"(time));
  return time;
}

inline unsigned Pds::SysClk::since (unsigned prev)
{
  return sample() - prev;
}

#else
inline double Pds::SysClk::nsPerTick()
{
  return 1.0;                           // 1 nS / "tick"
}

inline unsigned Pds::SysClk::sample()
{
  timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_nsec;
}

inline unsigned Pds::SysClk::since (unsigned prev)
{
  unsigned now = sample();
  return now > prev ? now - prev : now + 1000000000 - prev;
}

#endif

#endif

