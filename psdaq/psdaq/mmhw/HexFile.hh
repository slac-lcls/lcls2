#ifndef Pds_Mmhw_HexFile_hh
#define Pds_Mmhw_HexFile_hh

#include <stdint.h>
#include <vector>

namespace Pds {
  namespace Mmhw {
    class McsFile;
    class HexFile {
    public:
      HexFile(const char*);
      HexFile(const McsFile&, unsigned start=0);
    public:
      const std::vector<uint8_t>& data() const { return _dataList; }
    private:
      std::vector<uint8_t> _dataList;
    };
  }
}

#endif
