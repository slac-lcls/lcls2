#include "xtcdata/xtc/ShapesData.hh"

namespace XtcData
{
// must match enum in header file
static const int element_sizes[] = {
    sizeof(uint8_t),  // UINT8
    sizeof(uint16_t), // UINT16
    sizeof(uint32_t), // UINT32
    sizeof(uint64_t), // UINT64
    sizeof(int8_t),   // INT8
    sizeof(int16_t),  // INT16
    sizeof(int32_t),  // INT32
    sizeof(int64_t),  // INT64
    sizeof(float),    // FLOAT
    sizeof(double),   // DOUBLE
    sizeof(char),     // CHARSTR
    sizeof(int32_t),  // ENUMVAL
    sizeof(int32_t)   // ENUMDICT
};

const char* Name::str_type() // (DataType type)
{
  switch((DataType)_type) {
  case UINT8    : return std::move("UINT8");
  case UINT16   : return std::move("UINT16");
  case UINT32   : return std::move("UINT32");
  case UINT64   : return std::move("UINT64");
  case INT8     : return std::move("INT8");
  case INT16    : return std::move("INT16");
  case INT32    : return std::move("INT32");
  case INT64    : return std::move("INT64");
  case FLOAT    : return std::move("FLOAT");
  case DOUBLE   : return std::move("DOUBLE");
  case CHARSTR  : return std::move("CHARSTR");
  case ENUMVAL  : return std::move("ENUMVAL");
  case ENUMDICT : return std::move("ENUMDICT");
  };
  return nullptr;
}

int Name::get_element_size(Name::DataType type)
{
    return element_sizes[type];
};
};
