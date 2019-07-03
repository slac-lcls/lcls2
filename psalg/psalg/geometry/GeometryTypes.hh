#ifndef PSALG_GEOMETRYTYPES_H
#define PSALG_GEOMETRYTYPES_H

/** Usage
 *
 * #include "psalg/geometry/GeometryTypes.hh"
 */

//#include <cstddef>  // for std::size_t
#include <stdint.h>  // uint8_t, uint16_t, uint32_t, etc.

//-------------------

namespace geometry {

  /// Geometry types
  typedef double   pixel_coord_t; // pixel coordinate, size, etc.
  typedef double   pixel_area_t;  // arbitrary pixel area
  typedef uint32_t pixel_idx_t;   // pixel index along 1-d axis
  typedef uint16_t pixel_mask_t;  // mask
  typedef double   angle_t;       // angle degree or radian
  typedef unsigned gsize_t;       // size of array
  typedef unsigned segindex_t;    // segment index in parent detector

  /// Enumerator for X, Y, and Z axes
  enum AXIS {AXIS_X=0, AXIS_Y, AXIS_Z};

} // namespace geometry

//-------------------

#endif // PSALG_GEOMETRYTYPES_H

