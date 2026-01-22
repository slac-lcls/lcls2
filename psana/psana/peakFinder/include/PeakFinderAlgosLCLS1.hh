#ifndef PSALGOS_PEAKFINDERALGOSLCLS1_H
#define PSALGOS_PEAKFINDERALGOSLCLS1_H

// PeakFinderAlgos.h 2017-08-07
//-----------------------------

/* 2020-02-26 conversion to lcls2 by M.D.
 * copied from psalgos/include/PeakFinderAlgos.h and renamed as PeakFinderAlgosLCLS1.hh
 * namespace psalgos -> renamed to psalg1
 */

#include <string>
#include <vector>
#include <iostream> // for cout, ostream
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <cmath>    // for sqrt
#include <typeinfo> // for typeid
#include <iomanip>  // for std::setw

//#include "psalgos/Types.h"
//#include "psalgos/LocalExtrema.h"
#include "Types.hh"
#include "LocalExtrema.hh"

using namespace std;

namespace psalg1 {

typedef types::mask_t      mask_t;
typedef types::extrim_t    extrim_t;
typedef types::conmap_t    conmap_t;
typedef types::TwoIndexes  TwoIndexes;

/**
 * @brief Peak parameters
 */
//class Peak{
//public:

struct Peak{
  float seg;
  float row;
  float col;
  float npix;
  float npos;
  float amp_max;
  float amp_tot;
  float row_cgrav;
  float col_cgrav;
  float row_sigma;
  float col_sigma;
  float row_min;
  float row_max;
  float col_min;
  float col_max;
  float bkgd;
  float noise;
  float son;

  Peak(){} // do not fill out member by default
  /*
  Peak(const float& _seg      =0,
       const float& _row      =0,
       const float& _col      =0,
       const float& _npix     =0,
       const float& _npos     =0,
       const float& _amp_max  =0,
       const float& _amp_tot  =0,
       const float& _row_cgrav=0,
       const float& _col_cgrav=0,
       const float& _row_sigma=0,
       const float& _col_sigma=0,
       const float& _row_min  =0,
       const float& _row_max  =0,
       const float& _col_min  =0,
       const float& _col_max  =0,
       const float& _bkgd     =0,
       const float& _noise    =0,
       const float& _son      =0) :
       seg      (_seg      ),
       row      (_row      ),
       col      (_col      ),
       npix     (_npix     ),
       npos     (_npos     ),
       amp_max  (_amp_max  ),
       amp_tot  (_amp_tot  ),
       row_cgrav(_row_cgrav),
       col_cgrav(_col_cgrav),
       row_sigma(_row_sigma),
       col_sigma(_col_sigma),
       row_min  (_row_min  ),
       row_max  (_row_max  ),
       col_min  (_col_min  ),
       col_max  (_col_max  ),
       bkgd     (_bkgd     ),
       noise    (_noise    ),
       son      (_son      ){ }
  */

  //copy constructor http://en.cppreference.com/w/cpp/language/copy_constructor
  Peak(const Peak& o)
    : seg      (o.seg      )
    , row      (o.row      )
    , col      (o.col      )
    , npix     (o.npix     )
    , npos     (o.npos     )
    , amp_max  (o.amp_max  )
    , amp_tot  (o.amp_tot  )
    , row_cgrav(o.row_cgrav)
    , col_cgrav(o.col_cgrav)
    , row_sigma(o.row_sigma)
    , col_sigma(o.col_sigma)
    , row_min  (o.row_min  )
    , row_max  (o.row_max  )
    , col_min  (o.col_min  )
    , col_max  (o.col_max  )
    , bkgd     (o.bkgd     )
    , noise    (o.noise    )
    , son      (o.son      )
    {}

  Peak& operator=(const Peak& rhs) {
    seg         = rhs.seg;
    row         = rhs.row;
    col         = rhs.col;
    npix        = rhs.npix;
    npos        = rhs.npos;
    amp_max	= rhs.amp_max;
    amp_tot	= rhs.amp_tot;
    row_cgrav 	= rhs.row_cgrav;
    col_cgrav	= rhs.col_cgrav;
    row_sigma	= rhs.row_sigma;
    col_sigma	= rhs.col_sigma;
    row_min	= rhs.row_min;
    row_max	= rhs.row_max;
    col_min	= rhs.col_min;
    col_max	= rhs.col_max;
    bkgd	= rhs.bkgd;
    noise	= rhs.noise;
    son         = rhs.son;
    return *this;
  }
};

/// Stream insertion operator,
std::ostream&
operator<<(std::ostream& os, const Peak& p);

/**
 * @brief Structure to hold background algorithm results
 */

struct RingAvgRms {
  double   avg; // average intensity in the ring
  double   rms; // rms in the ring
  unsigned npx; // number of pixels used

  RingAvgRms(const double& av=0,
             const double& rm=0,
             const unsigned& np=0) :
    avg(av), rms(rm), npx(np) {}

  //copy constructor
  RingAvgRms(const RingAvgRms& o) : avg(o.avg), rms(o.rms), npx(o.npx){}

  RingAvgRms& operator=(const RingAvgRms& rhs) {
    avg = rhs.avg;
    rms = rhs.rms;
    npx = rhs.npx;
    return *this;
  }
};

std::ostream&
operator<<(std::ostream& os, const RingAvgRms& b);

/**
 *  @ingroup psalgos
 *
 *  @brief PeakFinderAlgos - class contains collection of methods for peak finding algorithms of 2-d image.
 *
 *  This software was developed for the LCLS project.
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @author Mikhail Dubrovin
 *
 *  @see ImgAlgos.ImgImgProc
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 *
 *  @li  Includes and typedefs
 *  @code
 *  #include <cstddef>  // for size_t
 *  #include "psalgos/PeakFinderAlgos.h"
 *  #include "psalgos/Types.h"
 *
 *  typedef types::mask_t     mask_t;
 *  typedef types::extrim_t   extrim_t;
 *  @endcode
 *
 *
 *  @li Define input parameters and make object
 *  \n
 *  @code
 *    const size_t rows = 1000;
 *    const size_t cols = 1000;
 *    const size_t rank = 5;
 *    const T      *data  = new T[rows*cols];
 *    const mask_t *mask  = new mask_t[rows*cols];
 *    extrim_t     *arr2d = new extrim_t[rows*cols];
 *
 *
 *    const float npix_min=1;
 *    const float npix_max=1e6;
 *    const float amax_thr=0;
 *    const float atot_thr=0;
 *    const float son_min=0;
 *
 *    const size_t seg=0;
 *    const unsigned& pbits=0 # Types.h:  NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16
 *    PeakFinderAlgos* alg = new PeakFinderAlgos(cseg, pbits);
 *  @endcode
 *
 *  @li Call methods
 *  \n
 *  @code
 *  alg.printParameters();
 *  alg.printMatrixOfRingIndexes();
 *  alg.printVectorOfRingIndexes();
 *  alg.setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min);
 *  alg.printSelectionPars();
 *
 *  alg.peakFinderV3r3<T>(data, mask, rows, cols, rank, r0, dr, nsigm);
 *
 *  const Peak& p = alg.peak(i)
 *  const Peak& p = alg.peakSelected(i)
 *  const std::vector<Peak> peaks = alg.vectorOfPeaks();
 *  const std::vector<Peak> peaks = alg.vectorOfPeaksSelected();
 *
 *  alg.localMaxima(arr2d, rows, cols);
 *  alg.localMinima(arr2d, rows, cols);
 *  alg.connectedPixels(arr2d, rows, cols);
 *
 *  @endcode
 */


class PeakFinderAlgos {
public:

  /**
   * @brief Class constructor is used for initialization of all paramaters.
   *
   * @param[in] seg    - ROI segment index in the ndarray
   * @param[in] pbits  - print control bit-word; =0-print nothing, =1 debug, =2 info, ...
   */

  //PeakFinderAlgos();
  PeakFinderAlgos(const size_t& seg=0, const unsigned& pbits=0);

  virtual ~PeakFinderAlgos();

  /// Prints memeber data
  void printParameters();

  /// Initialaise maps and vectors per event; uses m_img_size, m_npksmax, vectors and maps
  void _initMapsAndVectors();

  /// Evaluate ring indexes for S/N algorithm
  void _evaluateRingIndexes();

  /// Prints indexes for S/N algorithm
  void printMatrixOfRingIndexes();
  void printVectorOfRingIndexes();

  /// Recursive method finding connected pixels in constrained region and filling vector of indexes for V3r3.
  //template <typename T>
  //void _findConnectedPixelsInRegionVX(const T* data, const int& r, const int& c);

  /// Set peak selection parameters
  void setPeakSelectionPars(const float& npix_min=0
			   ,const float& npix_max=1e6
			   ,const float& amax_thr=0
			   ,const float& atot_thr=0
			   ,const float& son_min=0);

  /// Prints peak selection parameters
  void printSelectionPars();

  /// Decide if peak should be included or not in the output v_peaks
  bool _peakIsSelected(const Peak& peak);

  /// Makes vector of selected peaks v_peaks_sel from v_peaks
  void _makeVectorOfSelectedPeaks();

  /// Prints vector of peaks, i.e. v_peaks_sel or v_peaks
  void _printVectorOfPeaks(const std::vector<Peak>& v);

  /// Returns peak from v_peaks by specified index
  const Peak& peak(const int& i=0){return v_peaks[i];}

  /// Returns peak from v_peaks_sel by specified index
  const Peak& peakSelected(const int& i=0){return v_peaks_sel[i];}

  /// Returns vector of peaks v_peaks
  const std::vector<Peak>& vectorOfPeaks(){return v_peaks;}

  /// Returns vector of selected peaks v_peaks_sel
  const std::vector<Peak>& vectorOfPeaksSelected(){return v_peaks_sel;}

  /// Fills-out (returns) array of local maxima
  void localMaxima(extrim_t *map, const size_t& rows, const size_t& cols) {
    std::memcpy(map, m_local_maxima, rows*cols*sizeof(extrim_t));
  }

  /// Fills-out (returns) array of local minima
  void localMinima(extrim_t *map, const size_t& rows, const size_t& cols) {
    std::memcpy(map, m_local_minima, rows*cols*sizeof(extrim_t));
  }

  /// Fills-out (returns) array of m_conmap
  void connectedPixels(conmap_t *map, const size_t& rows, const size_t& cols) {
    std::memcpy(map, m_conmap, rows*cols*sizeof(conmap_t));
  }

private:
  size_t m_seg;      // segment index (for list of images)
  unsigned m_pbits;  // pirnt control bit-word
  float  m_r0;       // radial parameter of the ring for S/N evaluation algorithm
  float  m_dr;       // ring width for S/N evaluation algorithm
  size_t m_rows;     // number of rows
  size_t m_cols;     // number of cols
  size_t m_rank;     // rank of maximum for peakFinderV3
  float  m_nsigm;    // number of sigma to estimate background threshold
  size_t m_img_size; // size of the image
  size_t m_pixgrp_max_size; // size of droplet vector
  size_t m_npksmax;
  size_t m_nminima;
  conmap_t m_numreg;

  float m_thr_low;
  float m_thr_high;


  double m_reg_thr;  // threshold on intensity
  double m_reg_a0;   // intensity in the initial point for droplet
  int    m_reg_rmin; // region limit
  int    m_reg_rmax; // region limit
  int    m_reg_cmin; // region limit
  int    m_reg_cmax; // region limit

  extrim_t    *m_local_maxima;
  extrim_t    *m_local_minima;
  conmap_t    *m_conmap;
  const mask_t *m_mask;

  float  m_peak_npix_min; // peak selection parameter
  float  m_peak_npix_max; // peak selection parameter
  float  m_peak_amax_thr; // peak selection parameter
  float  m_peak_atot_thr; // peak selection parameter
  float  m_peak_son_min;  // peak selection parameter

  std::vector<TwoIndexes> v_ind_pixgrp; // vector of pixel indexes for droplet
  std::vector< std::vector<TwoIndexes> > vv_peak_pixinds; // vector of peak vector of pixel indexes
  std::vector<Peak> v_peaks;
  std::vector<Peak> v_peaks_sel;
  std::vector<TwoIndexes> v_indexes; // vector of indexes for background ring

  RingAvgRms m_bkgd;

public:

  /**
   * @brief peakFinderV3r3 - "Ranker" - further development of ImgAlgos.peakFinderV3r2.
   * Changes:
   *   - use packahe psalgos
   *   - get rid of ndarray
   *   - pass most of parameters via member data
   */

  //std::vector<Peak>&
template <typename T>
void
peakFinderV3r3(const T *data
              ,const mask_t *mask
              ,const size_t& rows
              ,const size_t& cols
              ,const size_t& rank
	      ,const double& r0=7.0
	      ,const double& dr=2.0
	      ,const double& nsigm=0)
{
  m_mask = mask;
  m_rows = rows;
  m_cols = cols;
  m_rank = rank;
  m_r0 = r0;
  m_dr = dr;
  m_nsigm = nsigm;
  m_img_size = rows*cols;
  m_pixgrp_max_size = (2*rank+1)*(2*rank+1);

  if(m_pbits & LOG::DEBUG) std::cout << "in peakFinderV3r3, rank=" << rank << '\n';
  if(m_pbits & LOG::INFO) printParameters();

  //std::cout << "XXX point 1 \n";
  //std::cout << "XXX typeid(T).name() << '\n';

  if (m_local_minima==0) m_local_minima = new extrim_t[m_img_size];
  if (m_local_maxima==0) m_local_maxima = new extrim_t[m_img_size];

  //std::fill_n(m_local_minima, int(m_img_size), extrim_t(0));
  //std::fill_n(m_local_maxima, int(m_img_size), extrim_t(0));

  m_nminima = localextrema::mapOfLocalMinimums<T>(data, mask, rows, cols, rank, m_local_minima); // fills m_local_minima
  m_npksmax = localextrema::mapOfLocalMaximums<T>(data, mask, rows, cols, rank, m_local_maxima); // fills m_local_maxima

  //-------
  // TEST OF numberOfExtrema and vectorOfExtremeIndexes
  //m_npksmax = localextrema::numberOfExtrema(m_local_maxima, rows, cols, 7);

  //std::cout << "XXX: nminima   =  " << m_nminima << '\n';
  //std::cout << "XXX: m_npksmax =  " << m_npksmax << '\n';

  //std::vector<TwoIndexes> v = localextrema::vectorOfExtremeIndexes(m_local_maxima, rows, cols, 7, m_npksmax);
  //std::cout << "vector<TwoIndexes> size: " << v.size() << '\n';
  //-------

  _initMapsAndVectors();
  _makeMapOfConnectedPixelsForLocalMaximums<T>(data); // fills m_conmap, v_peaks, vv_peak_pixinds
  _makeVectorOfSelectedPeaks();                       // make vector of selected peaks
  if (m_pbits & LOG::INFO) {
      _printVectorOfPeaks(v_peaks_sel);               // print vector of selected peaks
      std::cout << "  number of maxima         = " << m_npksmax << '\n';
      std::cout << "  number of minima         = " << localextrema::numberOfExtrema(m_local_minima, rows, cols, 7) << '\n';
      std::cout << "  number of found peaks    = " << v_peaks.size() << '\n';
      std::cout << "  number of selected peaks = " << v_peaks_sel.size() << '\n';
  }
  //return v_peaks_sel;
}

  /**
   * @brief full peak processing in a right order
   *          1. estimate background
   *          2. finds connected pixels
   *          3. fills vector of peaks
   *
   * @param[in]  data - pointer to 2-d array with calibrated intensities
   */
template <typename T>
void
_makeMapOfConnectedPixelsForLocalMaximums(const T *data)
{
  if(m_pbits & LOG::DEBUG) std::cout << "in _makeMapOfConnectedPixelsForLocalMaximums/n";

  const unsigned BIT_SEL=4; // <<=====

  int irc=0;
  m_numreg=0;
  for(int r=0; r<(int)m_rows; r++)
    for(int c=0; c<(int)m_cols; c++) {

        irc = r*m_cols+c;
        if(! (m_local_maxima[irc] & BIT_SEL)) continue;

        ++m_numreg;

        RingAvgRms bkgd = _evaluateRingAvgRmsV1<T>(data, r, c);
        m_reg_thr = bkgd.avg + m_nsigm * bkgd.rms; // <<=====
	//if (m_pbits & LOG::DEBUG)
	//  std::cout << "XXX: m_numreg=" << m_numreg << " r=" << std::setw(4) << std::setprecision(0) << r
        //                                            << " c=" << std::setw(4) << std::setprecision(0) << c
        //                                            << " bkgd:" << bkgd << " thr:" << m_reg_thr << '\n';

        _findConnectedPixelsForLocalMaximumV2<T>(data, r, c); // <<=====
	//std::cout << "XXX: number of connected pixels = " << v_ind_pixgrp.size() << '\n';

        if(v_ind_pixgrp.empty()) {
	  //std::cout << "XXX peakFinderV3r3 WARNING: v_ind_pixgrp is empty...\n";
          --m_numreg; continue;
        }

         vv_peak_pixinds.push_back(v_ind_pixgrp);

	_procPixGroupV1<T>(data, bkgd, v_ind_pixgrp); // proc connected group and fills v_peaks
    }
}

  /**
   * @brief Evaluate background average and rms in the ring around pixel using data, mask, and map of local intensity extremes.
   * development of ImgAlgos.Alg.imgProc._evaluateBkgdAvgRmsV3
   * _evaluateRingAvgRmsV1 news: get rid of ndarray, mask, m_r0, m_dr etc are passed as member data, RingAvgRms is created at return.
   *
   * Background average and rms are evaluated for any pixel specified by the (row,col).
   * Uses mask. Good non-extreme pixels are used.
   * This algorithm uses pixels in the ring m_r0, m_dr.
   *
   * @param[in]  data - pointer to 2-d array with calibrated intensities
   * @param[in]  row  - pixel row
   * @param[in]  col  - pixel column
   */

template <typename T>
RingAvgRms
_evaluateRingAvgRmsV1(const T *data
		     ,const int& row
                     ,const int& col
                     )
{
  //if(m_pbits & LOG::DEBUG) std::cout << "in _evaluateRingAvgRmsV1\n";

  double   amp  = 0;
  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;
  int      irc  = 0;

  for(vector<TwoIndexes>::const_iterator ij  = v_indexes.begin();
                                         ij != v_indexes.end(); ij++) {
    int ir = row + (ij->i);
    int ic = col + (ij->j);

    //std::cout << "YYY: " << " row=" << row << " col=" << col << " ir=" << ir << " ic=" << ic << '\n';

    if(ic < 0 || !(ic < (int)m_cols)) continue;
    if(ir < 0 || !(ir < (int)m_rows)) continue;

    // pixel selector:
    irc = ir*m_cols+ic;

    if(! m_mask[irc]) continue;           // skip masked pixels
    if(m_local_maxima[irc] & 7) continue; // skip extremal pixels
    if(m_local_minima[irc] & 7) continue; // skip extremal pixels

    amp = (double)data[irc];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }

  if(sum0) {
    sum1 /= sum0;                          // Averaged base level
    sum2 = sum2/sum0 - sum1*sum1;
    sum2 = (sum2>0) ? std::sqrt(sum2) : 0; // RMS of the background around peak
  }

  return RingAvgRms(sum1, sum2, sum0); // returns avg, rms, npx
}

  /** The same as _evaluateRingAvgRmsV1, but selection of pixel is for Droplet
   */

template <typename T>
RingAvgRms
_evaluateRingAvgRmsForDroplet(const T *data
		             ,const int& row
                             ,const int& col
                             )
{
  //if(m_pbits & LOG::DEBUG) std::cout << "in _evaluateRingAvgRmsForDroplet\n";

  double   amp  = 0;
  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;
  int      irc  = 0;

  for(vector<TwoIndexes>::const_iterator ij  = v_indexes.begin();
                                         ij != v_indexes.end(); ij++) {
    int ir = row + (ij->i);
    int ic = col + (ij->j);

    //std::cout << "YYY: " << " row=" << row << " col=" << col << " ir=" << ir << " ic=" << ic << '\n';

    if(ic < 0 || !(ic < (int)m_cols)) continue;
    if(ir < 0 || !(ir < (int)m_rows)) continue;

    // pixel selector:
    irc = ir*m_cols+ic;

    if(! m_mask[irc]) continue;               // skip masked pixels
    //if(! (m_local_maxima[irc] & 1)) continue; // skip pixels with |a| > thr_low
    if(! (m_local_maxima[irc] & 19)) continue; // skip pixels with |a| > thr_high

    amp = (double)data[irc];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }

  if(sum0) {
    sum1 /= sum0;                          // Averaged base level
    sum2 = sum2/sum0 - sum1*sum1;
    sum2 = (sum2>0) ? std::sqrt(sum2) : 0; // RMS of the background around peak
  }

  return RingAvgRms(sum1, sum2, sum0); // returns avg, rms, npx
}

  /**
   * @brief _findConnectedPixelsForLocalMaximumV2 - apply flood filling algorithms to find a group of connected pixels
   * around r0,c0 in constrained by rad region.
   *   - check that r0,c0 is absolute maximum, returns false if not found
   *   - apply flood filling and make vector of connected pixels in rad constrained by region.
   * V2 news: mask and rank are passed as member data
   *
   * @param[in]  data - pointer to 2-d array with calibrated intensities
   * @param[in]  r0 - droplet central pixel row-coordinate
   * @param[in]  c0 - droplet central pixel column-coordinate
   */

template <typename T>
void
_findConnectedPixelsForLocalMaximumV2(const T* data
                                     ,const int& r0
                                     ,const int& c0
                                     )
{
  // set group region limits
  m_reg_rmin = std::max(0,           int(r0-m_rank));
  m_reg_rmax = std::min((int)m_rows, int(r0+m_rank+1));
  m_reg_cmin = std::max(0,           int(c0-m_rank));
  m_reg_cmax = std::min((int)m_cols, int(c0+m_rank+1));

  //Clean rank-size m_conmap
  for(int r=m_reg_rmin; r<m_reg_rmax; r++)
    for(int c=m_reg_cmin; c<m_reg_cmax; c++) m_conmap[r*m_cols+c] = 0;

  //if(m_pbits & LOG::DEBUG) std::cout << "in _findConnectedPixelsForLocalMaximum, seg=" << m_seg
  //                          << " rank=" << m_rank  << " r0=" << r0 << " c0=" << c0 << '\n';
  //std::cout << "ZZZ _findConnectedPixelsForLocalMaximum : rank=" << rank  << " r0=" << r0 << " c0=" << c0 << '\n';
  //std::cout << "ZZZ: m_reg_rmin: " << m_reg_rmin << "  m_reg_rmax: " << m_reg_rmax
  //        << "  m_reg_cmin: " << m_reg_cmin << "  m_reg_cmax: " << m_reg_cmax << '\n';

  v_ind_pixgrp.clear();
  _findConnectedPixelsInRegionV3<T>(data, r0, c0); // begin recursion
}

// Templated recursive method finging connected pixels for lacalMaximums

template <typename T>
void
_findConnectedPixelsInRegionV3(const T* data, const int& r, const int& c)
{
  //if(m_pbits & LOG::DEBUG)
  //std::cout << "in _findConnectedPixelsInRegionV3, r=" << r << " c=" << c << '\n';
  int irc = r*m_cols+c;
  if(! m_mask[irc]) return; // - masked
  if(m_conmap[irc]) return; // - pixel is already used // ????????? double-counting or symmetry?
  if(data[irc] < (T)m_reg_thr) return; // discard pixel below threshold if m_reg_thr != 0

  m_conmap[irc] = m_numreg; // mark pixel on map

  //std::cout << " XXX r=" << r << " c=" << c << " m_numreg:" << m_numreg << '\n';

  v_ind_pixgrp.push_back(TwoIndexes(r,c));

  if(  r+1 < m_reg_rmax)  _findConnectedPixelsInRegionV3<T>(data, r+1, c);
  if(  c+1 < m_reg_cmax)  _findConnectedPixelsInRegionV3<T>(data, r, c+1);
  if(!(r-1 < m_reg_rmin)) _findConnectedPixelsInRegionV3<T>(data, r-1, c);
  if(!(c-1 < m_reg_cmin)) _findConnectedPixelsInRegionV3<T>(data, r, c-1);
}

  /**
   * @brief _procPixGroupV1 - process a pixel group for peakFinderV3r3
   * further development of ImgAlgos::AlgImgProc::_procPixGroup
   * V1 news: get rid of ndarray,
   *          use member data to pass parameters,
   *          subtract base level brfore processing
   *
   * @param[in] data  - pointer to data array with calibrated intensities
   * @param[in] bkgd  - structure of base level ave, rms, npix
   * @param[in] vinds - vector of connected pixel indexes
   */

template <typename T>
void
_procPixGroupV1(const T* data
	       ,const RingAvgRms& bkgd
	       ,const std::vector<TwoIndexes>& vinds
               )
{
  if(vinds.empty()) return;

  const int& r0 = vinds[0].i;
  const int& c0 = vinds[0].j;

  //if(m_pbits & LOG::DEBUG) std::cout << "in _procPixGroup r0=" << r0 << " c0=" << c0 << '\n';

  int irc0 = r0*m_cols+c0;

  double   a0 = data[irc0] - bkgd.avg;
  unsigned npix = 0;
  double   samp = 0;
  double   sac1 = 0;
  double   sac2 = 0;
  double   sar1 = 0;
  double   sar2 = 0;
  int      rmin = m_rows;
  int      cmin = m_cols;
  int      rmax = 0;
  int      cmax = 0;

  for(vector<TwoIndexes>::const_iterator ij  = vinds.begin();
                                         ij != vinds.end(); ij++) {
      int r = ij->i;
      int c = ij->j;
      int irc = r*m_cols+c;
      double a = data[irc] - bkgd.avg;

      if(r<rmin) rmin=r;
      if(r>rmax) rmax=r;
      if(c<cmin) cmin=c;
      if(c>cmax) cmax=c;

      npix += 1;
      samp += a;
      sar1 += a*r;
      sac1 += a*c;
      sar2 += a*r*r;
      sac2 += a*c*c;
  }

  if(npix<1) return;

  Peak peak;

  peak.seg     = m_seg;
  peak.row     = r0;
  peak.col     = c0;
  peak.npix    = npix;
  peak.npos    = 0;
  peak.amp_max = a0; // - bkgd.avg;
  peak.amp_tot = samp; // - bkgd.avg * npix;

  if(samp>0) {
    sar1 /= samp;
    sac1 /= samp;
    sar2 = sar2/samp - sar1*sar1;
    sac2 = sac2/samp - sac1*sac1;
    peak.row_cgrav = sar1;
    peak.col_cgrav = sac1;
    peak.row_sigma = (npix>1 && sar2>0) ? std::sqrt(sar2) : 0;
    peak.col_sigma = (npix>1 && sac2>0) ? std::sqrt(sac2) : 0;
  }
  else {
    peak.row_cgrav = r0;
    peak.col_cgrav = c0;
    peak.row_sigma = 0;
    peak.col_sigma = 0;
  }

  peak.row_min   = rmin;
  peak.row_max   = rmax;
  peak.col_min   = cmin;
  peak.col_max   = cmax;
  peak.bkgd      = bkgd.avg;
  peak.noise     = bkgd.rms;
  //peak.bkgnpx   = bkgd.npx;
  double noise_tot = bkgd.rms * std::sqrt(npix);
  peak.son       = (noise_tot>0) ? peak.amp_tot / noise_tot : 0;

  v_peaks.push_back(peak);
}

  /**
   * @brief peakFinderV4r3 - further development of ImgAlgos.peakFinderV4r2.
   * Changes:
   *   - use packahe psalgos
   *   - get rid of ndarray
   *   - pass most of parameters via member data - "Droplet-finder" - further improvement of V4r1.
   */
template <typename T>
void
peakFinderV4r3(const T *data
              ,const mask_t *mask
              ,const size_t& rows
              ,const size_t& cols
              ,const double& thr_low
              ,const double& thr_high
              ,const size_t& rank=5
	      ,const double& r0=7.0
	      ,const double& dr=2.0
	       //,const double& nsigm=0
              )
{
  m_mask = mask;
  m_rows = rows;
  m_cols = cols;
  m_thr_low  = thr_low;
  m_thr_high = thr_high;
  m_rank = rank;
  m_r0 = r0;
  m_dr = dr;
  m_nsigm = 0; // nsigm;
  m_img_size = rows*cols;
  m_pixgrp_max_size = (2*rank+1)*(2*rank+1);

  if(m_pbits & LOG::DEBUG) std::cout << "in peakFinderV4r3, rank=" << rank << '\n';
  if(m_pbits & LOG::INFO) printParameters();

  if (m_local_minima==0) m_local_minima = new extrim_t[m_img_size];
  if (m_local_maxima==0) m_local_maxima = new extrim_t[m_img_size];
  std::fill_n(m_local_minima, int(m_img_size), extrim_t(0));

  //m_nminima = localextrema::mapOfLocalMinimums<T>(data, mask, rows, cols, rank, m_local_minima);
  m_npksmax = localextrema::mapOfThresholdMaximums<T>(data ,mask, rows, cols, rank, thr_low, thr_high, m_local_maxima);

  //std::cout << "XXX: nminima   =  " << m_nminima << '\n';
  //std::cout << "XXX: m_npksmax =  " << m_npksmax << '\n';

  _initMapsAndVectors();
  _makeMapOfConnectedPixelsForDroplets<T>(data);
  _makeVectorOfSelectedPeaks();
  //std::cout << "XXX: number of connected pixels = " << v_ind_pixgrp.size() << '\n';

}

  /**
   * @brief full peak processing in a right order
   *          1. estimate background
   *          2. finds connected pixels
   *          3. fills vector of peaks
   *
   * @param[in]  data - pointer to 2-d array with calibrated intensities
   */
template <typename T>
void
_makeMapOfConnectedPixelsForDroplets(const T *data)
{
  if(m_pbits & LOG::DEBUG) std::cout << "in _makeMapOfConnectedPixelsForDroplets/n";

  const unsigned BIT_SEL=8; // <<===== bit for maximaa in rank in mapOfThresholdMaximums

  int irc=0;
  m_numreg=0;
  for(int r=0; r<(int)m_rows; r++)
    for(int c=0; c<(int)m_cols; c++) {

        irc = r*m_cols+c;
        if(! (m_local_maxima[irc] & BIT_SEL)) continue;

        ++m_numreg;

        RingAvgRms bkgd = _evaluateRingAvgRmsForDroplet<T>(data, r, c);
        m_reg_thr = m_thr_low; // bkgd.avg + m_nsigm * bkgd.rms; // <<=====
	//if (m_pbits & LOG::DEBUG)
	//  std::cout << "XXX: m_numreg=" << m_numreg << " r=" << std::setw(4) << std::setprecision(0) << r
        //                                            << " c=" << std::setw(4) << std::setprecision(0) << c
        //                                            << " bkgd:" << bkgd << " thr:" << m_reg_thr << '\n';

        _findConnectedPixelsForDroplet(r, c); // <<=====
	//std::cout << "XXX A: number of connected pixels = " << v_ind_pixgrp.size() << '\n';

        if(v_ind_pixgrp.empty()) {
	  //std::cout << "XXX peakFinderV3r3 WARNING: v_ind_pixgrp is empty...\n";
          --m_numreg; continue;
        }

         vv_peak_pixinds.push_back(v_ind_pixgrp);

	_procPixGroupV1<T>(data, bkgd, v_ind_pixgrp); // proc connected group and fills v_peaks
    }
}

  /**
   * @brief _findConnectedPixelsForDroplet - apply flood filling algorithms to find a group of connected pixels
   * news: rank is passed as member data
   *
   * @param[in]  r0 - droplet central pixel row-coordinate
   * @param[in]  c0 - droplet central pixel column-coordinate
   */

//template <typename T>
void
_findConnectedPixelsForDroplet(const int& r0
                              ,const int& c0
                              )
{
  m_reg_rmin = std::max(0,           int(r0-m_rank));
  m_reg_rmax = std::min((int)m_rows, int(r0+m_rank+1));
  m_reg_cmin = std::max(0,           int(c0-m_rank));
  m_reg_cmax = std::min((int)m_cols, int(c0+m_rank+1));

  //Clean rank-size m_conmap
  for(int r=m_reg_rmin; r<m_reg_rmax; r++)
    for(int c=m_reg_cmin; c<m_reg_cmax; c++) m_conmap[r*m_cols+c] = 0;

  //if(m_pbits & LOG::DEBUG) std::cout << "in _findConnectedPixelsForLocalMaximum, seg=" << m_seg
  //                                   << " rank=" << m_rank  << " r0=" << r0 << " c0=" << c0 << '\n';

  v_ind_pixgrp.clear();
  _findConnectedPixelsInDroplet(r0, c0); // begin recursion
}

// Recursive method finging connected pixels for Droplet

void
_findConnectedPixelsInDroplet(const int& r, const int& c)
{
  //if(m_pbits & LOG::DEBUG)
  //std::cout << "in _findConnectedPixelsInDroplet, r=" << r << " c=" << c << '\n';
  int irc = r*m_cols+c;
  if(m_local_maxima[irc]<2) return; // 0: masked, 1: <thr_low
  if(m_conmap[irc]) return; // - pixel is already used

  m_conmap[irc] = m_numreg; // mark pixel on map

  v_ind_pixgrp.push_back(TwoIndexes(r,c));

  if(  r+1 < m_reg_rmax)  _findConnectedPixelsInDroplet(r+1, c);
  if(  c+1 < m_reg_cmax)  _findConnectedPixelsInDroplet(r, c+1);
  if(!(r-1 < m_reg_rmin)) _findConnectedPixelsInDroplet(r-1, c);
  if(!(c-1 < m_reg_cmin)) _findConnectedPixelsInDroplet(r, c-1);
}

}; // class PeakFinderAlgos
} // namespace psalg1

#endif // PSALGOS_PEAKFINDERALGOSLCLS1_H

