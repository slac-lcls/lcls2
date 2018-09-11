#ifndef McsFile_hh
#define McsFile_hh

#include <vector>

class McsFile {
public:
  McsFile(const char*);
public:
  unsigned startAddr () { return _startAddr; }
  unsigned endAddr   () { return _endAddr; }
  unsigned write_size() { return _write_size; }
  unsigned read_size () { return _read_size; }
  void     entry(unsigned i,
                 unsigned& addr,
                 unsigned& data)
  { addr = _addrList[i]; data = _dataList[i]; }
private:
  unsigned _startAddr;  
  unsigned _endAddr;
  unsigned _write_size;
  unsigned _read_size;
  std::vector<unsigned> _addrList;
  std::vector<unsigned> _dataList;
};

#endif
