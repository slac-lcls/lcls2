#ifndef Pds_Mmhw_McsFile_hh
#define Pds_Mmhw_McsFile_hh

#include <vector>

namespace Pds {
  namespace Mmhw {
    class McsFile {
    public:
      McsFile(const char*);
    public:
      unsigned startAddr () const { return _startAddr; }
      unsigned endAddr   () const { return _endAddr; }
      unsigned write_size() const { return _write_size; }
      unsigned read_size () const { return _read_size; }
      void     entry(unsigned i,
                     unsigned& addr,
                     unsigned& data) const
      { addr = _addrList[i]; data = _dataList[i]; }
    private:
      unsigned _startAddr;  
      unsigned _endAddr;
      unsigned _write_size;
      unsigned _read_size;
      std::vector<unsigned> _addrList;
      std::vector<unsigned> _dataList;
    };
  }
}

#endif
