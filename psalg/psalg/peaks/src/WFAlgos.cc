
#include "psalg/peaks/WFAlgos.hh"

#include <list>
#include <utility>  // pair
//#include <iostream> // cout

namespace psalg {

static void 
_add_edge(NDArray<const double> v,
          bool                  rising, // leading positive or trailing negative edge
	  double                fraction_value,
          double                deadtime,
	  double                peak, 
	  unsigned              start, 
	  double&               last,
	  std::list< std::pair<double,double> >&   result)
{
  // find the edge
  double edge_v = fraction_value;
  unsigned i=start;
  if (rising) {
    while(v(i) < edge_v)
      i++;
  }
  else { // trailing positive edge, or leading negative edge
    while(v(i) > edge_v)
      i++;
  }
  double edge = i>0 ? 
    (edge_v-v(i))/(v(i)-v(i-1))
    + double(i) : 0;

  if (last < 0 || edge > last + deadtime) {
    //cout << "XXX add peak intensity:" << peak << " edge index:" << edge << '\n';
    std::pair<double,double> a(peak, edge);
    result.push_back(a);
    last = edge;
  }
}


//find leading or trailing edges
NDArray<double>*
find_edges(NDArray<const double>& wf,
           double                 baseline_value,
           double                 threshold_value,
           double                 fraction,
           double                 deadtime,
           bool                   leading_edge)
{
  std::list< std::pair<double,double> > result;
  double   peak   = threshold_value;
  unsigned start  = 0;
  double   last   = -deadtime-1.0;
  bool     rising = threshold_value > baseline_value;
  bool     crossed=false;
  for(unsigned k=0; k<wf.shape()[0]; k++) {
    double y = wf(k);
    bool over = 
      ( rising && y>threshold_value) ||
      (!rising && y<threshold_value);
    if (!crossed && over) {
      crossed = true;
      start   = k;
      peak    = y;
    }
    else if (crossed && !over) {
      _add_edge(wf, rising==leading_edge,
                fraction*(peak+baseline_value),
                deadtime,
                peak, start, last, result);
      crossed = false;
    }
    else if (( rising && y>peak) ||
             (!rising && y<peak)) {
      peak = y;
      if (!leading_edge) // for a trailing edge, start at the peak!
        start = k;
    }
  }
    
  // the last edge may not have fallen back below threshold
  if (crossed) {
    _add_edge(wf, rising==leading_edge,
              fraction*(peak+baseline_value),
              deadtime,
              peak, start, last, result);
  }

  types::shape_t shape[] = {(types::shape_t)result.size(),2};
  NDArray<double>* p_edges = new NDArray<double>(shape, 2);
  NDArray<double>& edges = *p_edges;

  unsigned k=0;
  for(std::list< std::pair<double,double> >::iterator it=result.begin();
     it!=result.end(); it++,k++) {
     edges(k,0) = (*it).first;
     edges(k,1) = (*it).second;
  }
  //std::cout << "XXX edges: " << edges << '\n';
  return p_edges;
}

} // namespace psalg
