/**
 * @file
 * @brief Declares psalg::DirFileIterator, which iterates over directory entries whose names contain a pattern.
 */
#ifndef PSALG_DIRFILEITERATOR_H
#define PSALG_DIRFILEITERATOR_H

//---------------------------------------------------
// Created on 2018-07-17 by Mikhail Dubrovin
//---------------------------------------------------

/** Usage
 *
 *  #include "psalg/utils/DirFileIterator.hh"
 *
 */

#include <string>
#include <dirent.h> // opendir, readdir, closedir, dirent, DIR

// #include "psalg/utils/Logger.hh" // MSG
// MSG(INFO, "In test_readdir");

//-------------------

//using namespace std;

namespace psalg {

//-------------------

/** Iterates over the entries of a directory whose names contain a pattern; next() returns dirname + "/" + name. */
class DirFileIterator {

public:

  /** Open dirname with opendir() (a WARNING is logged if that fails) and keep pattern (null means every entry). Only the pointers are kept, and the default dirname is a hard-coded /reg/neh/home/dubrovin path. */
  DirFileIterator(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work/",
                  const char* pattern=0); // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
  /** Close the directory with closedir(), without checking that it was opened. */
  ~DirFileIterator();

  /**
   * Return the next entry ("." and ".." included) whose name contains the pattern, as dirname + "/" + name, or an empty string when none is left.
   * The returned reference refers to a member string overwritten by the next call; a directory that failed to open is not checked.
   */
  const std::string& next();

private:

  const char* _dirname;
  const char* _pattern;
  DIR*        _dir;
  dirent*     _pdir;
  std::string _fname;
  const std::string _empty_string;

}; //class DirFileIterator

 //-------------------

} // namespace psalg

#endif // PSALG_DIRFILEITERATOR_H
