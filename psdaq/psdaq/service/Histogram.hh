/*
** ++
**  Package:
**	Utility
**
**  Abstract:
**
**  Author:
**      Michael Huffer, SLAC, (650) 926-4269
**
**  Creation Date:
**	000 - October 27,1999
**
**  Revision History:
**	None.
**
** --
*/

#ifndef PDS_HISTOGRAM
#define PDS_HISTOGRAM

#include <stdint.h>

namespace Pds {
class Histogram
  {
  public:
    Histogram(unsigned size, double unitsCvt);
   ~Histogram();
  public:
    void     sum();
    void     dump(char* filesSpec);
    void     dump() const;
    double   units()     const;
    double   weight()    const;
    double   counts()    const;
    unsigned overflows() const;
    void     bump(uint64_t index);
    void     reset();
  private:
    unsigned* _buffer;        // Histogram buffer
    unsigned  _oflow;         // Bin for overflows
    unsigned  _mask;          // Control overflows
    unsigned  _size;          // Number of entries
    uint64_t  _maxIdx;        // Maximum index seen
    double    _totalCounts;   // # of times histogram incrmented
    double    _totalWeight;   // # of times histogram incrmented
    double    _unitsCvt;
  };
}
/*
** ++
**
**
** --
*/

inline double Pds::Histogram::units() const
  {
  return _unitsCvt;
  }

/*
** ++
**
**
** --
*/

inline double Pds::Histogram::counts() const
  {
  return _totalCounts;
  }

/*
** ++
**
**
** --
*/

inline double Pds::Histogram::weight() const
  {
  return _totalWeight;
  }

/*
** ++
**
**
** --
*/

inline unsigned Pds::Histogram::overflows() const
  {
  return _oflow;
  }

/*
** ++
**
**
** --
*/

inline void Pds::Histogram::bump(uint64_t index)
  {
  unsigned* buffer = _buffer;
  if (index < _size)
    buffer[index]++;
  else
    _oflow++;
  if (index > _maxIdx)  _maxIdx = index;
  }

#endif
