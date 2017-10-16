#include "xtcdata/xtc/ShapesData.hh"

// must match enum in header file
static const int element_sizes[] = { sizeof(uint8_t), sizeof(uint16_t), sizeof(int32_t),
                                     sizeof(float), sizeof(double) };

int Name::get_element_size(Name::DataType type)
{
    return element_sizes[type];
}
