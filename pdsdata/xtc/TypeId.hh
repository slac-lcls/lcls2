#ifndef Pds_TypeId_hh
#define Pds_TypeId_hh

#include <stdint.h>

namespace Pds {

  class TypeId {
  public:
    /*
     * Notice: New enum values should be appended to the end of the enum list, since
     *   the old values have already been recorded in the existing xtc files.
     */
    enum Type {
      Parent,
      Data,
      Desc,
      NumberOf};
    enum { VCompressed = 0x8000 };

    TypeId() {}
    TypeId(const TypeId& v);
    TypeId(Type type, uint32_t version, bool compressed=false);
    TypeId(const char*);

    Type     id()      const;
    uint32_t version() const;
    uint32_t value()   const;

    bool     compressed() const;

    static const char* name(Type type);
    static uint32_t _sizeof() { return sizeof(TypeId); }

  private:
    uint32_t _value;
  };

}

#endif
