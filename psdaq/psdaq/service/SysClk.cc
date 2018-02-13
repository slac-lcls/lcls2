#include "psdaq/service/SysClk.hh"

using namespace Pds;

long long int SysClk::diff(const timespec& end, 
			   const timespec& start) 
{
  long long int diff;
  diff =  (end.tv_sec - start.tv_sec);
  if (diff) diff *= 1000000000LL;
  diff += end.tv_nsec;
  diff -= start.tv_nsec;
  return diff;
}
