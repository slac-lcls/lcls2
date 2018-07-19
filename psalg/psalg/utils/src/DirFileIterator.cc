
//#include <iostream> // cout, puts etc.
#include "psalg/utils/Logger.hh" // MSG
#include "psalg/utils/DirFileIterator.hh"

//#include <experimental/filesystem>
//typedef std::experimental::filesystem::path fspath;

//-------------------

using namespace std;
//using namespace psalg;

namespace psalg {

//-------------------

  DirFileIterator::DirFileIterator(const char* dirname, const char* pattern)
    : _dirname(dirname)
    , _pattern(pattern) {

    if((_dir = opendir(dirname)) == NULL) {
      MSG(WARNING, "Cannot open directory: " << _dirname);
    } else { 
      MSG(DEBUG, "c-tor open directory: " << _dirname);
    }
  }

//-------------------

  DirFileIterator::~DirFileIterator() {
    MSG(DEBUG, "d-tor close directory: " << _dirname);
    closedir(_dir);
  }

//-------------------
  // Returns next file name found in the directory and consistent with pattern or empty string.
  const string& DirFileIterator::next() {

    while ((_pdir = readdir(_dir)) != NULL) {
      _fname = _pdir->d_name;
      if(_pattern && _fname.find(_pattern) == std::string::npos) continue;
      //MSG(DEBUG, "  -- next file: " << _fname);
      //fspath dir(_dirname);
      //fspath file(_fname);
      //_fname = std::string(dir / file);
      _fname = std::string(_dirname) + '/' + _fname;
      return _fname;
    }
    return _empty_string;
  }

//-------------------

} // namespace psalg

