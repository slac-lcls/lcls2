#ifndef PSALG_SEGGEOMETRYSTORE_H
#define PSALG_SEGGEOMETRYSTORE_H

//------------------

#include <string>

#include "psalg/geometry/SegGeometryCspad2x1V1.hh"
#include "psalg/geometry/SegGeometryEpix100V1.hh"
#include "psalg/geometry/SegGeometryEpix10kaV1.hh"
#include "psalg/geometry/SegGeometryMatrixV1.hh"

//#include "MsgLogger/MsgLogger.h"
#include "psalg/utils/Logger.hh" // MSG, LOGGER

//-------------------

namespace psalg {

/// @addtogroup psalg psalg

/**
 *  @ingroup psalg
 *
 *  @brief class SegGeometryStore has a static factory method Create for SegGeometry object
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see SegGeometry, SegGeometryCspad2x1V1
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Includes
 *  @code
 *  // #include "psalg/geometry/SegGeometry.hh" // already included under SegGeometryStore.h
 *  #include "psalg/geometry/SegGeometryStore.hh"
 *  typedef psalg::SegGeometry SG;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Classes like SegGeometryCspad2x1V1 containing implementation of the SegGeometry interface methods are self-sufficient. 
 *  Factory method Create should returns the pointer to the SegGeometry object by specified segname parameter or returns 0-pointer if segname is not recognized (and not implemented).
 *  Code below instateates SegGeometry object using factory static method psalg::SegGeometryStore::Create()
 *  @code
 *  std::string source = "SENS2X1:V1";
 *  or 
 *  std::string source = "EPIX100:V1";
 *  or 
 *  std::string source = "PNCCD:V1";
 *  unsigned print_bits=0377; // does not print by default if parameter omited
 *  psalg::SegGeometry* segeo = psalg::SegGeometryStore::Create(segname, print_bits);
 *  @endcode
 *
 *  @li Print info
 *  @code
 *  segeo -> print_seg_info(0377);
 *  @endcode
 *
 *  @li Access methods
 *  \n are defined in the interface SegGeometry and implemented in SegGeometryCspad2x1V1
 *  @code
 *  // scalar values
 *  const SG::size_t         array_size        = segeo -> size(); 
 *  const SG::size_t         number_of_rows    = segeo -> rows();
 *  const SG::size_t         number_of_cols    = segeo -> cols();
 *  const SG::pixel_coord_t  pixel_scale_size  = segeo -> pixel_scale_size();
 *  const SG::pixel_coord_t  pixel_coord_min   = segeo -> pixel_coord_min(SG::AXIS_Z);
 *  const SG::pixel_coord_t  pixel_coord_max   = segeo -> pixel_coord_max(SG::AXIS_X);
 * 
 *  // pointer to arrays with size equal to array_size
 *  const SG::size_t*        p_array_shape     = segeo -> shape();
 *  const SG::pixel_area_t*  p_pixel_area      = segeo -> pixel_area_array();
 *  const SG::pixel_coord_t* p_pixel_size_arr  = segeo -> pixel_size_array(SG::AXIS_X);
 *  const SG::pixel_coord_t* p_pixel_coord_arr = segeo -> pixel_coord_array(SG::AXIS_Y);
 *  @endcode
 *
 *  @li How to add new segment to the factory
 *  \n 1. implement SegGeometry interface methods in class like SegGeometryCspad2x1V1
 *  \n 2. add it to SegGeometryStore with unique segname
 */

//----------------

class SegGeometryStore  {
public:

  //SegGeometryStore () {}
  //virtual ~SegGeometryStore () {}

  /**
   *  @brief Static factory method for SegGeometry of the segments defined by the name
   *  
   *  @param[in] segname        segment name
   *  @param[in] print_bits     print control bit-word.
   */ 

  static psalg::SegGeometry*
  Create (const std::string& segname="SENS2X1:V1", const unsigned print_bits=0)
  {
      //if (print_bits & 1) MsgLog("SegGeometryStore", info, "Segment geometry factory for " << segname);
        if (print_bits & 1) MSG(INFO, "Segment geometry factory for " << segname);
        if (segname=="SENS2X1:V1")  { return psalg::SegGeometryCspad2x1V1::instance(); } // use singleton
        if (segname=="EPIX100:V1")  { return psalg::SegGeometryEpix100V1::instance(); } // use singleton
        if (segname=="EPIX10KA:V1") { return psalg::SegGeometryEpix10kaV1::instance(); } // use singleton
        if (segname=="PNCCD:V1")    { return new psalg::SegGeometryMatrixV1(512,512,75.,75.,400.,75.); }
        if (segname.find("MTRX") != std::string::npos) { 

          MSG(INFO, "Segment geometry factory for " << segname);
          std::size_t rows;
	  std::size_t cols;
	  float   rpixsize;
	  float   cpixsize;

	  if(! psalg::matrix_pars(segname, rows, cols, rpixsize, cpixsize)) {
            if (print_bits) MSG(ERROR, "Can't demangle geometry segment name: " << segname);  
	    return 0; // NULL;
	  }

	  if (print_bits & 1) MSG(INFO, "segname: " << segname
                                      << " rows: " << rows << " cols:" << cols 
                                      << " rpixsize: " << rpixsize << " cpixsize: " << cpixsize);

          return new psalg::SegGeometryMatrixV1(rows, cols, rpixsize, cpixsize); 
                                               // pix_size_depth, pix_scale_size); 
        }

        if (print_bits & 2) MSG(INFO, "Segment geometry is undefined for segment name " << segname << " - return 0-pointer...");  
        //abort();
	return 0; // NULL;
  }
};

} // namespace psalg

#endif // PSALG_SEGGEOMETRYSTORE_H
