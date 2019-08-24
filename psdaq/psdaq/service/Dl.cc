#include "Dl.hh"

#include <string.h>                     // strdup
#include <stdio.h>                      // fprintf


int Pds::Dl::open(const std::string& filename, int flag)
{
  if (_handle)
  {
    fprintf(stderr, "%s:\n  Cannot open dynamic library '%s' without first closing '%s'\n",
            __PRETTY_FUNCTION__, filename.c_str(), _filename.c_str());
    return -1;
  }

  _filename = filename;

  _handle = dlopen(_filename.c_str(), flag);
  if (!_handle)
  {
    fprintf(stderr, "%s:\n  Cannot open dynamic library '%s':\n  %s\n",
            __PRETTY_FUNCTION__, _filename.c_str(), dlerror());
    return -1;
  }

  return 0;
}

void* Pds::Dl::loadSymbol(const std::string& name) const
{
  dlerror();                            // Reset errors

  void*       symbol = dlsym(_handle, name.c_str());
  const char *error  = dlerror();
  if (error)
  {
    fprintf(stderr, "%s:\n  Cannot load symbol '%s' from '%s':\n  %s\n",
            __PRETTY_FUNCTION__, name.c_str(), _filename.c_str(), error);
    return nullptr;
  }

  return symbol;
}

void Pds::Dl::close()
{
  if (_handle)
  {
    dlclose(_handle);
    _handle = nullptr;
  }
}
