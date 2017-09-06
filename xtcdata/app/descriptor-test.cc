#include "pdsdata/xtc/Descriptor.hh"

int main()
{
    int num_fields = 3;

    int desc_size = sizeof(int) + num_fields * sizeof(Field);
    uint8_t header[desc_size];
    *reinterpret_cast<int*>(header) = num_fields;

    Descriptor desc(header);

    Field* field = desc.get_field_by_index(0);
    strncpy(field->name, "field1", 256);
    field->type = FLOAT;
    field->offset = 0;

    field = desc.get_field_by_index(1);
    strncpy(field->name, "field2", 256);
    field->type = FLOAT_ARRAY;
    field->offset = 4;
    field->rank = 2;
    field->shape = { 1024, 1024 };

    field = desc.get_field_by_index(2);
    strncpy(field->name, "field3", 256);
    field->type = INT;
    field->offset = 1024 * 1024 + 4;

    field = desc.get_field_by_name("field2");
    std::cout << field->name << std::endl;

    /*
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::cout<<std::chrono::duration_cast<std::chrono::nanoseconds>(end -
    start).count() / double(n)<<"  ns per call"<<std::endl;
    */
}
