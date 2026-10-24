/**
 * @file
 * @brief Dl, a small wrapper around dlopen(), dlsym() and dlclose().
 */
#ifndef Pds_Dl_hh
#define Pds_Dl_hh

#include <string>
#include <dlfcn.h>                      // dlopen, flag definitions, etc.

namespace Pds
{
  /** Owns one dynamic library handle opened with dlopen(). */
  class Dl
  {
  public:
    /** Construct with no library open. */
    Dl() : _handle(nullptr) { }
    /** Call close(). */
    ~Dl() { close(); }
  public:
    /** Open filename with dlopen() and flag; returns 0, or -1 (logged at debug level) if a library is already open or dlopen() fails. */
    int   open(const std::string& filename, int flag);
    /** Close the library with dlclose() if one is open. */
    void  close();
    /** Return the address of symbol name from the open library, or nullptr (logged at debug level) if dlsym() reports an error. */
    void* loadSymbol(const std::string& name) const;
  private:
    std::string _filename;
    void*       _handle;
  };
}

#endif
