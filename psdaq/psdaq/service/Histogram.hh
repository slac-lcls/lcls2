/**
 * @file
 * @brief Histogram, a power-of-2 sized integer histogram with an overflow count.
 */
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
/** Integer histogram with 2^size bins and an overflow count; unitsCvt converts a bin index to physical units when printing. */
class Histogram
  {
  public:
    /** Allocate 2^size zeroed bins (size is the log2 of the bin count) and keep the units conversion unitsCvt. */
    Histogram(unsigned size, double unitsCvt);
   /** Free the bins. */
   ~Histogram();
  public:
    /** Recompute the total counts and the count-weighted sum of bin indices from the bins. */
    void     sum();
    /** Write the non-empty bins (index times units, count), the overflow count, and the totals, mean and maximum index to the file filesSpec; prints a message if the file cannot be opened or closed. */
    void     dump(char* filesSpec);
    /** Print all bins, the overflow count, and the totals, mean and maximum index to stdout (calls sum()). */
    void     dump() const;
    /** Return the units conversion factor per bin. */
    double   units()     const;
    /** Return the count-weighted sum of bin indices from the last sum(). */
    double   weight()    const;
    /** Return the total counts from the last sum(). */
    double   counts()    const;
    /** Return the number of indices that were beyond the last bin. */
    unsigned overflows() const;
    /** Increment the bin index, or the overflow count if index is beyond the last bin, and track the largest index seen. */
    void     bump(uint64_t index);
    /** Zero all bins, the overflow count, the largest index and the totals. */
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
