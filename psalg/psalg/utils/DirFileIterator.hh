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

class DirFileIterator {

public:

  DirFileIterator(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work/",
                  const char* pattern=0); // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
  ~DirFileIterator();

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
