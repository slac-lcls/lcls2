#include "pdsdata/xtc/Descriptor.hh"

Descriptor::Descriptor() : num_fields(0) {}

// Field * Descriptor::get_field_by_name(const char* name)
// {
//      Field* field = nullptr;
//      for (int i=0; i<_num_fields; i++) {
//          field = get_field_by_index(i);
//          if (!strcmp(field->name, name)) {
//             return field;
//         }
//     }
//     // name not found in descriptor
//     std::cout<<"name not found in desc"<<std::endl;
//     return nullptr;
// }
