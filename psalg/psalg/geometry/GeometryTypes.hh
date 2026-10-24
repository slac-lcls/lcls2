/**
 * @file
 * @brief Declares the basic geometry:: type aliases (coordinates, areas, indexes, masks, sizes) and the AXIS enum.
 */
#ifndef PSALG_GEOMETRYTYPES_H
#define PSALG_GEOMETRYTYPES_H

/** Usage
 *
 * #include "psalg/geometry/GeometryTypes.hh"
 */

//#include <cstddef> // for std::size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, etc.

//-------------------

namespace geometry {

  /// Geometry types
  typedef double   pixel_coord_t; // pixel coordinate, size, etc.
  /** double; the trailing comment calls it an arbitrary pixel area. */
  typedef double   pixel_area_t;  // arbitrary pixel area
  /** uint32_t; per the trailing comment, a pixel index along a 1-d axis. */
  typedef uint32_t pixel_idx_t;   // pixel index along 1-d axis
  /** uint16_t; mask element type. */
  typedef uint16_t pixel_mask_t;  // mask
  /** double; the trailing comment says it holds an angle in degrees or radians. */
  typedef double   angle_t;       // angle degree or radian
  /** unsigned; the trailing comment says it is an array size. */
  typedef unsigned gsize_t;       // size of array
  /** unsigned; per the trailing comment, a segment index in the parent detector. */
  typedef unsigned segindex_t;    // segment index in parent detector
  /** unsigned; per the trailing comment, a bit word that controls mask layers, printed bits, etc. */
  typedef unsigned bitword_t;     // bitword to control mask layers, print bits etc.

  /// Enumerator for X, Y, and Z axes
  enum AXIS {AXIS_X=0, /**< Value 0. */ AXIS_Y, /**< Value 1. */ AXIS_Z /**< Value 2. */ };

} // namespace geometry

//-------------------

#endif // PSALG_GEOMETRYTYPES_H

