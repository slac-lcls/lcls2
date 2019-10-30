#ifndef PSALG_UTILS_H
#define PSALG_UTILS_H

//---------------------------------------------------
// Created on 2018-07-17 by Mikhail Dubrovin
//---------------------------------------------------

/** Usage
 *
 *  #include "psalg/utils/Utils.hh"
 *
 */

#include <vector>
#include <string>
#include <iostream> // cout, puts etc.
#include <dirent.h> // opendir, readdir, closedir, dirent, DIR

#include <algorithm> // sort

// #include "psalg/utils/Logger.hh" // MSG
// MSG(INFO, "In test_readdir");

//-------------------

using namespace std;

namespace psalg {

//-------------------

    std::vector<std::string>  
    files_in_dir(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work",
                 const char* pattern=0) { // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
        dirent* pdir;
        DIR* dir = opendir(dirname);
	const std::string str_dir(dirname);
	std::string fname;
	std::vector<std::string> v_files;

        while ((pdir = readdir(dir)) != NULL) {
	  fname = pdir->d_name;
	  if(pattern && fname.find(pattern) == std::string::npos) continue;
          //cout << "  XXX:files_in_dir: " << fname << endl;
          v_files.push_back(str_dir + '/' + fname);
        }
        closedir(dir);
        sort(v_files.begin(),v_files.end()); 
        return v_files;
    }

//-------------------

    void dir_content(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work",
                     const char* pattern=0) { // "nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1") {
        dirent* pdir;
        DIR* dir = opendir(dirname);
        while ((pdir = readdir(dir)) != NULL) {
	  std::string fname(pdir->d_name);
	  if(pattern && fname.find(pattern) == std::string::npos) continue;
          cout << fname << endl;
        }
        closedir(dir);
    }

//---------

/*
template <typename T1, typename T2>
struct Pair{
  T1 val1;
  T2 val2;
  Pair(const T1& v1=0, const T2& v2=0) : val1(v1), val2(v2) {}
  Pair(const Pair& o) : val1(o.val1), val2(o.val2) {}
  Pair& operator=(const Pair& rhs) {val1 = rhs.val1; val2 = rhs.val2; return *this;}
};
*/

//---------

} // namespace psalg

#endif // PSALG_UTILS_H
