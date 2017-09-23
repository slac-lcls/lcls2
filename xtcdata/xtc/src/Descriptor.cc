#include "xtcdata/xtc/Descriptor.hh"

Field* Desc::get_field_by_name(const char* name)
{
    for (unsigned i = 0; i < num_fields(); i++) {
        Field* field = &get(i);
        if (!strcmp(field->name, name)) {
            return field;
        }
    }
    // name not found in descriptor
    std::cout << "name not found in desc" << std::endl;
    return nullptr;
}
