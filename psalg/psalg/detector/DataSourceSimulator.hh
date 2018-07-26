#ifndef PSALG_DATASOURCESIMULATOR_H
#define PSALG_DATASOURCESIMULATOR_H
//-----------------------------

#include <string>
//#include "psalg/calib/NDArray.hh" // NDArray
//#include "psalg/detector/Types.hh"

#include "psalg/utils/DirFileIterator.hh"
#include "psalg/detector/AreaDetectorStore.hh"

//using namespace std;
//using namespace psalg;

namespace detector {

//-----------------------------
class DataSourceSimulator {
public:

  DataSourceSimulator(const char* dirname="/reg/neh/home/dubrovin/LCLS/con-detector/work/",
                      const char* pattern=0) // pattern="nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1"
    : _file_iterator(dirname, pattern) {}

  virtual ~DataSourceSimulator() {}

  

  //virtual const size_t   size (const event_t&);

  //virtual const NDArray<raw_t>& raw(const event_t&);

private:
  //std::string _detname;
  //NDArray<raw_t> _raw_nda;
    DirFileIterator _file_iterator;

}; // class

} // namespace detector

#endif // PSALG_DATASOURCESIMULATOR_H
//-----------------------------
