#ifndef HSD_HISTOGRAM
#define HSD_HISTOGRAM

namespace HSD {
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
    void     bump(unsigned index);
    void     reset();
  private:
    unsigned* _buffer;        // Histogram buffer
    unsigned  _mask;          // Control overflows
    unsigned  _size;          // Number of entries
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

inline double HSD::Histogram::units() const
  {
  return _unitsCvt;
  }

/*
** ++
**
**
** --
*/

inline double HSD::Histogram::counts() const
  {
  return _totalCounts;
  }

/*
** ++
**
**
** --
*/

inline double HSD::Histogram::weight() const
  {
  return _totalWeight;
  }

/*
** ++
**
**
** --
*/

inline unsigned HSD::Histogram::overflows() const
  {
  return *_buffer;
  }

/*
** ++
**
**
** --
*/

inline void HSD::Histogram::bump(unsigned index)
  {
  unsigned* buffer = _buffer;
  if (index < _size)
    buffer[index]++;
  else
    (*buffer)++;
  }

#endif
