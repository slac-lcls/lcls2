/**
 * @file
 * @brief Declares the psalg::types aliases shape_t and size_t.
 */
#ifndef PSALG_TYPES_H
#define PSALG_TYPES_H

//---------------------------------------------------
// Created on 2018-07-12 by Mikhail Dubrovin
//---------------------------------------------------

namespace psalg {
  /** Basic psalg type aliases. */
  namespace types {

    /** uint32_t; the trailing comment points to xtcdata/xtc/Array.hh. */
    typedef uint32_t shape_t; // see xtcdata/xtc/Array.hh
    /** uint32_t. Inside psalg::types it hides the standard size_t. */
    typedef uint32_t size_t;

  } // namespace types
} // namespace psalg

#endif // PSALG_TYPES_H
