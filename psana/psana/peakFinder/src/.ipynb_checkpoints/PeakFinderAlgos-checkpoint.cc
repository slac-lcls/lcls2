//-----------------------------

#include "../include/PeakFinderAlgos.hh"
#include <sstream>   // for stringstream
#include <cmath>     // floor, ceil
#include <iomanip>   // for std::typedef

//-----------------------------

using namespace std;

//-----------------------------

namespace psalgos {

//-----------------------------

PeakFinderAlgos::PeakFinderAlgos(Allocator *allocator, const size_t& seg, const unsigned& pbits, const size_t& lim_rank, const size_t& lim_peaks)
  : m_seg(seg)
  , m_pbits(pbits)
  , m_rank(lim_rank)
  , m_pixgrp_max_size((2*m_rank+1)*(2*m_rank+1))
  , m_npksmax(lim_peaks)
  , m_local_maxima(0)
  , m_local_minima(0)
  , m_conmap(0)
  , m_peak_npix_min(0)
  , m_peak_npix_max(1e6)
  , m_peak_amax_thr(0)
  , m_peak_atot_thr(0)
  , m_peak_son_min(0)
  , arr_ind_pixgrp_drp(allocator, m_pixgrp_max_size)
  , aa_peak_pixinds_drp(allocator, lim_peaks)
  , arr_peaks_drp(allocator, lim_peaks)
  , arr_peaks_sel_drp(allocator, lim_peaks)
  , arr_indexes_drp(allocator, m_pixgrp_max_size)
  , m_allocator(allocator)
  , rows(allocator, 0)
  , cols(allocator, 0)
  , intens(allocator, 0)
{
  #ifndef NDEBUG
  if(m_pbits & LOG::DEBUG) std::cout << "in c-tor PeakFinderAlgos\n";
  #endif
}

//-----------------------------

PeakFinderAlgos::~PeakFinderAlgos() 
{
  #ifndef NDEBUG
  if(m_pbits & LOG::DEBUG) std::cout << "in d-tor ~PeakFinderAlgos\n";
  #endif
  if (m_local_maxima) delete[] m_local_maxima;
  if (m_local_minima) delete[] m_local_minima;
  if (m_conmap)       delete[] m_conmap;
}

void
PeakFinderAlgos::setAllocator(Allocator *allocator) {
    m_allocator = allocator;
}

//-----------------------------

void
PeakFinderAlgos::printParameters()
{
  std::stringstream ss; 
  ss << "PeakFinderAlgos::printParameters\n";
  ss << "seg   " << m_seg << '\n';
  ss << "pbits " << m_pbits << '\n';
  ss << "rows  " << m_rows << '\n';
  ss << "cols  " << m_cols << '\n';
  ss << "rank  " << m_rank << '\n';
  ss << "r0    " << m_r0 << '\n';
  ss << "dr    " << m_dr << '\n';
  ss << "nsigm " << m_nsigm << '\n';
  ss << "img_size " << m_img_size << '\n';
  ss << "sizeof(extrim_t) " << sizeof(extrim_t) << '\n';
  cout << ss.str();
}

//-----------------------------

void PeakFinderAlgos::_convPeaksSelected(){
    numPeaksSelected = arr_peaks_sel_drp.num_elem();
    rows = AllocArray1D<float>(m_allocator, numPeaksSelected);
    cols = AllocArray1D<float>(m_allocator, numPeaksSelected);
    intens = AllocArray1D<float>(m_allocator, numPeaksSelected);
    for(unsigned i = 0; i< numPeaksSelected; i++){
        const Peak p = arr_peaks_sel_drp(i);
        rows.push_back(p.row_cgrav);
        cols.push_back(p.col_cgrav);
        intens.push_back(p.amp_tot);
    }
}


void 
PeakFinderAlgos::_initMapsAndVectors_drp()
{
  #ifndef NDEBUG
  if(m_pbits & LOG::DEBUG) std::cout << "in _initMapsAndVectors_drp\n";
  #endif

  if (m_conmap==0) m_conmap = new conmap_t[m_img_size];

  std::fill_n(m_conmap, int(m_img_size), conmap_t(0));

  arr_ind_pixgrp_drp.clear();
  aa_peak_pixinds_drp.clear();
  arr_peaks_drp.clear();

  _evaluateRingIndexes_drp();

}

//-----------------------------

void 
PeakFinderAlgos::_evaluateRingIndexes_drp()
{
  #ifndef NDEBUG
  if(m_pbits & LOG::DEBUG) std::cout << "in _evaluateRingIndexes_drp, r0=" << m_r0 << " dr=" << m_dr << '\n';
  #endif

  int indmax = (int)std::ceil(m_r0 + m_dr);
  int indmin = -indmax;

  arr_indexes_drp.clear();

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {
      double r = std::sqrt(double(i*i + j*j));
      if (r < m_r0 || r > m_r0 + m_dr) continue;
      arr_indexes_drp.push_back(TwoIndexes(i,j));
    }
  }

  #ifndef NDEBUG
  if(m_pbits) {
    printMatrixOfRingIndexes();
    printVectorOfRingIndexes_drp();
  }
  #endif
}

//-----------------------------

void 
PeakFinderAlgos::printMatrixOfRingIndexes()
{
  int indmax = (int)std::ceil(m_r0 + m_dr);
  int indmin = -indmax;
  unsigned counter = 0;
  std::stringstream ss; 
  
  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {
      double r = std::sqrt(double(i*i + j*j));
      int status = (r < m_r0 || r > m_r0 + m_dr) ? 0 : 1;
      if (status) counter++;
      if (i==0 && j==0) ss << " +";
      else              ss << " " << status;
    }
    ss << '\n';
  }
  ss << "Number of pixels to estimate background = " << counter << '\n';
  cout << ss.str();
}

//-----------------------------

void 
PeakFinderAlgos::printVectorOfRingIndexes_drp()
{
  if(arr_indexes_drp.num_elem() == 0) {
    _evaluateRingIndexes_drp();
  }

  std::stringstream ss; 
  ss << "In printVectorOfRingIndexes:\n Vector size: " << arr_indexes_drp.num_elem() << '\n';
  int counter_in_line=0;
  for (unsigned int ii = 0; ii < arr_indexes_drp.num_elem(); ii++) {
    ss << " (" << arr_indexes_drp(ii).i << "," << arr_indexes_drp(ii).j << ')';
    if (++counter_in_line > 9) {ss << '\n'; counter_in_line=0;}
  }   
  cout << ss.str() << '\n';
}
//-----------------------------

void
PeakFinderAlgos::printSelectionPars()
{
  std::stringstream ss; 
  ss << "PeakFinderAlgos::printSelectionPars(), seg=" << m_seg << '\n';
  ss << "  npix_min" << m_peak_npix_min << '\n';
  ss << "  npix_max" << m_peak_npix_max << '\n';
  ss << "  amax_thr" << m_peak_amax_thr << '\n';
  ss << "  atot_thr" << m_peak_atot_thr << '\n';
  ss << "  son_min " << m_peak_son_min  << '\n';
  cout << ss.str();
}

//-----------------------------

void
PeakFinderAlgos::setPeakSelectionPars(const float& npix_min, const float& npix_max,
                                      const float& amax_thr, const float& atot_thr, const float& son_min)
{
  #ifndef NDEBUG
  if(m_pbits & LOG::DEBUG) cout << "in setPeakSelectionPars, seg=" << m_seg << '\n';
  #endif

  m_peak_npix_min = npix_min;
  m_peak_npix_max = npix_max;
  m_peak_amax_thr = amax_thr;
  m_peak_atot_thr = atot_thr;
  m_peak_son_min  = son_min;
}

//-----------------------------

bool
PeakFinderAlgos::_peakIsSelected(const Peak& peak)
{
  if (peak.son     < m_peak_son_min)  return false;
  if (peak.npix    < m_peak_npix_min) return false;
  if (peak.npix    > m_peak_npix_max) return false;
  if (peak.amp_max < m_peak_amax_thr) return false;
  if (peak.amp_tot < m_peak_atot_thr) return false;
  return true;
}

//-----------------------------

void
PeakFinderAlgos::_makeVectorOfSelectedPeaks_drp()
{
  arr_peaks_sel_drp.clear();

  for(unsigned int ii = 0; ii < arr_peaks_drp.num_elem(); ii++) {
    Peak peak = arr_peaks_drp(ii);

    if(_peakIsSelected(peak)) {
        arr_peaks_sel_drp.push_back(peak);
    }
  }

  #ifndef NDEBUG
  if(m_pbits) std::cout << "_makeVectorOfSelectedPeaks, seg=" << m_seg 
                        << "  #peaks raw=" << arr_peaks_drp.num_elem()
                        << "  sel=" << arr_peaks_sel_drp.num_elem() << '\n';
  #endif
}


//-----------------------------

void
PeakFinderAlgos::_printVectorOfPeaks_drp(AllocArray1D<Peak> v) {
  for(unsigned int ii = 0; ii < v.num_elem(); ii++) {
    const Peak p = v(ii);
    std::cout << fixed
       << "Seg:"      << std::setw(3) << std::setprecision(0) << p.seg
       << " Row:"     << std::setw(4) << std::setprecision(0) << p.row
       << " Col:"     << std::setw(4) << std::setprecision(0) << p.col
       << " Npix:"    << std::setw(3) << std::setprecision(0) << p.npix
       << " Imax:"    << std::setw(7) << std::setprecision(1) << p.amp_max
       << " Itot:"    << std::setw(7) << std::setprecision(1) << p.amp_tot
       << " CGrav r:" << std::setw(6) << std::setprecision(1) << p.row_cgrav
       << " c:"       << std::setw(6) << std::setprecision(1) << p.col_cgrav
       << " Sigma r:" << std::setw(5) << std::setprecision(2) << p.row_sigma
       << " c:"       << std::setw(5) << std::setprecision(2) << p.col_sigma
       << " Rows["    << std::setw(4) << std::setprecision(0) << p.row_min
       << ":"         << std::setw(4) << std::setprecision(0) << p.row_max
       << "] Cols["   << std::setw(4) << std::setprecision(0) << p.col_min
       << ":"         << std::setw(4) << std::setprecision(0) << p.col_max
       << "] B:"      << std::setw(5) << std::setprecision(1) << p.bkgd
       << " N:"       << std::setw(5) << std::setprecision(1) << p.noise
       << " S/N:"     << std::setw(5) << std::setprecision(1) << p.son
       << std::endl;
  }
}

//-----------------------------
//-- NON-CLASS METHODS
//-----------------------------

  std::ostream& 
  operator<<(std::ostream& os, const Peak& p) 
  {
    os << fixed
       << "Seg:"      << std::setw(3) << std::setprecision(0) << p.seg
       << " Row:"     << std::setw(4) << std::setprecision(0) << p.row 	     
       << " Col:"     << std::setw(4) << std::setprecision(0) << p.col 	      
       << " Npix:"    << std::setw(3) << std::setprecision(0) << p.npix    
       << " Imax:"    << std::setw(7) << std::setprecision(1) << p.amp_max     	      
       << " Itot:"    << std::setw(7) << std::setprecision(1) << p.amp_tot    	      
       << " CGrav r:" << std::setw(6) << std::setprecision(1) << p.row_cgrav 	      
       << " c:"       << std::setw(6) << std::setprecision(1) << p.col_cgrav   	      
       << " Sigma r:" << std::setw(5) << std::setprecision(2) << p.row_sigma  	      
       << " c:"       << std::setw(5) << std::setprecision(2) << p.col_sigma  	      
       << " Rows["    << std::setw(4) << std::setprecision(0) << p.row_min    	      
       << ":"         << std::setw(4) << std::setprecision(0) << p.row_max    	      
       << "] Cols["   << std::setw(4) << std::setprecision(0) << p.col_min    	      
       << ":"         << std::setw(4) << std::setprecision(0) << p.col_max    	     
       << "] B:"      << std::setw(5) << std::setprecision(1) << p.bkgd       	      
       << " N:"       << std::setw(5) << std::setprecision(1) << p.noise      	     
       << " S/N:"     << std::setw(5) << std::setprecision(1) << p.son;
    return os;
  }

//-----------------------------

  std::ostream& 
  operator<<(std::ostream& os, const RingAvgRms& o) 
  {
    os << fixed
       << " Bkgd avg:" << std::setw(7) << std::setprecision(1) << o.avg
       << " RMS:"      << std::setw(7) << std::setprecision(1) << o.rms
       << " Npix:"     << std::setw(4) << std::setprecision(0) << o.npx;
    return os;
  }

//-----------------------------
} // namespace psalgos
//-----------------------------
