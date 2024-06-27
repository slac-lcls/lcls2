
#include <iostream> // cout
#include "../include/WFAlgos.hh"
//#include "psana/peakFinder/WFAlgos.hh"

//#include <list>
//#include <cstddef>  // size_t
//#include <utility>  // pair

namespace psalg {

template <typename T>
void
_add_edge(
  const std::vector<T>& v,
  bool     rising, // leading positive or trailing negative edge
  double   fraction,
  double   deadtime,
  T        peak,
  index_t  start,
  double&  last,
  index_t& ipk,
  T*       pkvals,
  index_t* pkinds)
{
  /*
    std::cout << "In WFAlgos.cc - _add_edge input parameters: "
              << " ipk:" << ipk
              << " rising:" << rising
              << " fraction:" << fraction
              << " deadtime:" << deadtime
              << " peak:" << peak
              << " start:" << start
              << " last:" << last
              << '\n';
  */

  // find the edge
  double edge_v = fraction;
  index_t i=start;
  if (rising) {
    while(v[i] < edge_v)
      i++;
  }
  else { // trailing positive edge, or leading negative edge
    while(v[i] > edge_v)
      i++;
  }
  double edge = i>0 ?
    (edge_v-v[i])/(v[i]-v[i-1])
    + double(i) : 0;

  if (last < 0 || edge > last + deadtime) {
    //cout << "XXX add peak intensity:" << peak << " edge index:" << edge << '\n';
    pkvals[ipk] = peak;
    pkinds[ipk] = (index_t)edge;
    ipk ++;
    last = edge;
  }
}


//find leading or trailing edges
template <typename T>
index_t
find_edges(
  index_t  npkmax,
  T*       pkvals,
  index_t* pkinds,
  const std::vector<T>& wf,
  double   baseline_f8,
  double   threshold_f8,
  double   fraction,
  double   deadtime,
  bool     leading_edge)
{
  //std::cout << "In WFAlgos.cc - find_edges wf: ";
  //for(typename std::vector<T>::const_iterator i=wf.begin(); i<wf.end(); ++i) std::cout << *i << ' ';
  /*
    std::cout << "In WFAlgos.cc - find_edges input parameters: "
              << " baseline:" << baseline_f8
              << " threshold:" << threshold_f8
              << " fraction:" << fraction
              << " deadtime:" << deadtime
              << " leading_edge:" << leading_edge
              << " wf.size():" << wf.size()
              << '\n';
  */

  T        baseline = (T)baseline_f8;   // because cython.... does not accept templeted/fused type...
  T        threshold = (T)threshold_f8;
  T        peak   = threshold;
  double   last   = -deadtime-1.0;
  bool     rising = threshold > baseline;
  bool     crossed= false;
  index_t  npk    = 0;

  index_t  start  = 0;
  index_t  k=0;
  for(; k<wf.size(); k++) {
    T y = wf[k];
    bool over =
       (rising && y>threshold) ||
      (!rising && y<threshold);

    if(!crossed && over) {
      crossed = true;
      start   = k;
      peak    = y;
    }
    else if(crossed && !over) {
      // add peak if its width exceeds deadtime
      if(double(k-start)>deadtime)
        _add_edge(wf, rising==leading_edge, fraction*(peak-baseline)+baseline,
                     deadtime, peak, start, last, npk, pkvals, pkinds);
      crossed = false;
      if(!(npk < npkmax)) break;
    }
    else if(( rising && y>peak) ||
            (!rising && y<peak)) {
      peak = y;
      if(!leading_edge) // for a trailing edge, start at the peak!
        start = k;
    }
  }

  // the last edge may not have fallen back below threshold
  if(crossed && (npk < npkmax) && (double(k-start)>deadtime)) {
    _add_edge(wf, rising==leading_edge, fraction*(peak-baseline)+baseline,
                 deadtime, peak, start, last, npk, pkvals, pkinds);
  }

  //std::cout << "\nIn WFAlgos.cc - find_edges found npk: " << npk << '\n';
  //std::cout << "\n  - pkinds: "; for(index_t i=0; i<npk; ++i) std::cout << pkinds[i] << ' ';
  //std::cout << "\n  - pkvals: "; for(index_t i=0; i<npk; ++i) std::cout << pkvals[i] << ' ';
  //std::cout << '\n';

  return npk;
}

#ifdef INST_FIND_EDGES
#undef INST_FIND_EDGES
#endif
#define INST_FIND_EDGES(T)\
  template index_t find_edges<T>\
    (index_t,T*,index_t*,const std::vector<T>&,double,double,double,double,bool);

INST_FIND_EDGES(double)
INST_FIND_EDGES(float)
INST_FIND_EDGES(int)
INST_FIND_EDGES(int64_t)
INST_FIND_EDGES(int16_t)
//INST_FIND_EDGES(int32_t)// the same as int

} // namespace psalg
