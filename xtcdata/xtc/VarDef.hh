#ifndef VARDEF__H
#define VARDEF__H

#include <vector>
class VarDef
{
public:
  struct detElem
  {
    const char* name;
    int type;
    int rank;
  };
  std::vector <detElem> detVec;
  enum DataType { UINT8, UINT16, INT32, FLOAT, DOUBLE, UINT64 };
  };
#endif  // VARDEF__H
