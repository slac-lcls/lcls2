#ifndef Pds_Dl_hh
#define Pds_Dl_hh

#include <string>
#include <dlfcn.h>                      // dlopen, flag definitions, etc.

namespace Pds
{
  class Dl
  {
  public:
    Dl() : _filename({}), _handle(nullptr) { }
    ~Dl() { close(); }
  public:
    int   open(const std::string& filename, int flag);
    void  close();
    void* loadSymbol(const std::string& name) const;
  private:
    std::string _filename;
    void*       _handle;
  };
};

#endif
