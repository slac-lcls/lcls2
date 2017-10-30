#include <stdio.h>
#include "Histogram.hh"

using namespace HSD;

/*
** ++
**
**   constructor... initializes all bins to zero.
**
** --
*/

Histogram::Histogram(unsigned size, double unitsCvt)
  {
  unsigned remaining = 1 << size;

  _buffer   = new unsigned[remaining];
  _size     = remaining;
  _mask     = remaining - 1;
  _unitsCvt = unitsCvt;

  reset();
  }


/*
** ++
**
**
** --
*/

void Histogram::reset()
  {
  unsigned  remaining = _size;
  unsigned* next      = _buffer;
  do *next++ = 0; while(--remaining);

  _totalWeight = 0.0;
  _totalCounts = 0.0;
  }

/*
** ++
**
**    Bump the bin for each client represented by the input argument.
**    This argument is a bit-list of the clients whose entry are to be
**    incremented.
**
** --
*/

void Histogram::sum()
  {
  unsigned  remaining   = _size;
  unsigned* next        = &_buffer[remaining];
  double    totalWeight = 0;
  double    totalCounts = 0;

  remaining--;

  do
    {
    double counts = (double) *--next;
    totalWeight += counts*(double)remaining;
    totalCounts += counts;
    }
  while(--remaining);

  _totalWeight = totalWeight;
  _totalCounts = totalCounts;
  }

/*
** ++
**
**    Bump the bin for each client represented by the input argument.
**    This argument is a bit-list of the clients whose entry are to be
**    incremented.
**
** --
*/

void Histogram::dump(char* fileSpec)
  {
  unsigned  remaining   = _size;
  unsigned* next        = &_buffer[remaining];

  remaining--;

  FILE* file = fopen (fileSpec, "w");
  if (!file)
    {
    printf ("Histogram::dump: Couldn't open file %s for write\n", fileSpec);
    return;
    }

  do
    {
    unsigned counts = *--next;
    if (counts)
      {
      fprintf (file, "%f %u\n", (float) remaining * _unitsCvt, counts);
      }
    }
  while(remaining--);

  if (fclose(file) == -1)
    {
    printf ("Histogram::dump: File %s didn't properly close\n", fileSpec);
    }
  }

void Histogram::dump() const
{
  printf("_size [%u]\n",_size);
  for(unsigned i=0; i<_size; i++)
    printf("%10u%c", _buffer[i], (i%10)==9 ? '\n':' ');
  printf("\n------------\n");
  const_cast<Histogram*>(this)->sum();
  printf("mean %f\n",_totalWeight/_totalCounts);
}

/*
** ++
**
**
** --
*/

Histogram::~Histogram()
  {
  delete[] _buffer;
  }
