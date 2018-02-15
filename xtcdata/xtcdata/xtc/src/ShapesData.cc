#include "xtcdata/xtc/ShapesData.hh"

namespace XtcData
{
// must match enum in header file
static const int element_sizes[] = { sizeof(uint8_t), sizeof(uint16_t), sizeof(int32_t),
                                     sizeof(float), sizeof(double), sizeof(uint64_t),
                                     sizeof(uint32_t), sizeof(int8_t), sizeof(int16_t),
                                     sizeof(int64_t) };

int Name::get_element_size(Name::DataType type)
{
    return element_sizes[type];
};
};
