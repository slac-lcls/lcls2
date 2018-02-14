/*
** ++
**  Package:
**	Utility
**
**  Abstract:
**	Non-inline functions for "Histogram.hh"
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

#include <stdio.h>
#include "Histogram.hh"

using namespace Pds;

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
  _maxIdx   = 0;

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
  _oflow  = 0;
  _maxIdx = 0;

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

  FILE* file = fopen (fileSpec, "w");
  if (!file)
    {
    printf ("Histogram::dump: Couldn't open file %s for write\n", fileSpec);
    return;
    }

  fprintf(file, "# _size [%u], bin width [%f]\n\n",_size, _unitsCvt);

  if (_oflow)  fprintf (file, "%f %u\n", double(remaining) * _unitsCvt, _oflow);
  --remaining;
  do
    {
    unsigned counts = *--next;
    if (counts)
      {
      fprintf (file, "%f %u\n", double(remaining) * _unitsCvt, counts);
      }
    }
  while(remaining--);

  const_cast<Histogram*>(this)->sum();
  fprintf(file, "\n# entries %f\n",_totalCounts);
  fprintf(file,   "# oflows  %u\n",_oflow);
  fprintf(file,   "# mean    %f\n",_unitsCvt*_totalWeight/_totalCounts);
  fprintf(file,   "# max     %f (idx %lu)\n",_unitsCvt*_maxIdx, _maxIdx);

  if (fclose(file) == -1)
    {
    printf ("Histogram::dump: File %s didn't properly close\n", fileSpec);
    }
  }

void Histogram::dump() const
{
  printf("_size [%u], bin width [%f]\n",_size, _unitsCvt);
  for(unsigned i=0; i<_size; i++)
    printf("%10u%c", _buffer[i], (i%10)==9 ? '\n':' ');
  if (_oflow)  printf("%10u", _oflow);
  printf("\n------------\n");
  const_cast<Histogram*>(this)->sum();
  printf("entries %f\n",_totalCounts);
  printf("oflows  %u\n",_oflow);
  printf("mean    %f\n",_unitsCvt*_totalWeight/_totalCounts);
  printf("max     %f (idx %lu)\n",_unitsCvt*_maxIdx, _maxIdx);
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
