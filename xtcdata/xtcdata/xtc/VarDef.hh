/**
 * @file
 * @brief Declares XtcData::VarDef, a list of Name definitions.
 */

#ifndef VARDEF__H
#define VARDEF__H

#include <vector>




namespace XtcData
{
  class Name;  

  /** Holder for a list of Name definitions; Names::add() appends a copy of each entry in NameVec to a Names xtc. */
  class VarDef
{
public:
  std::vector <Name> NameVec;  ///< Name definitions, in order.
  };

};
#endif  // VARDEF__H
