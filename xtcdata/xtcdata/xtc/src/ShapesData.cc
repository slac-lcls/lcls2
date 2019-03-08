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
    sizeof(uint32_t), // ENUMVAL
    sizeof(uint32_t)  // ENUMDICT
};

int Name::get_element_size(Name::DataType type)
{
    return element_sizes[type];
};
};
