#include "pdsdata/xtc/Descriptor.hh"

int get_element_size(Type& type) 
{
  const static int element_sizes[] = {
      1, // UINT8,
      2, // UINT16,
      4, // INT32,
      4, // FLOAT,
      8  // DOUBLE
    };
    return element_sizes[type];
}

Descriptor::Descriptor() : num_fields(0)
{
}

Field* Descriptor::get_field_by_name(const char* name)
{
    for(int i = 0; i < num_fields; i++) {
        Field* field = &get(i);
        if(!strcmp(field->name, name)) {
            return field;
        }
    }
    // name not found in descriptor
    std::cout << "name not found in desc" << std::endl;
    return nullptr;
}
