#ifndef Pds_Trg_utilities_hh
#define Pds_Trg_utilities_hh

#include "psdaq/service/Dl.hh"
#include "psdaq/service/json.hpp"

#include <cstdint>
#include <string>

#include "rapidjson/document.h"


namespace Pds {
  namespace Trg {

    int fetchDocument(const std::string&   connectMsg,
                      const std::string&   configAlias,
                      const std::string&   section,
                      rapidjson::Document& top);

    template <typename T>
    class Factory
    {
    public:
      Factory() : _object(nullptr) {}
    public:
      T* create(const rapidjson::Document& top,
                const std::string&         detName,
                const std::string&         symbol);
    private:
      typedef T* Create_t();
      Pds::Dl _dl;
      T*      _object;
    };
  };
};


template <typename T>
T* Pds::Trg::Factory<T>::create(const rapidjson::Document& top,
                                const std::string&         detName,
                                const std::string&         symbol)
{
  using namespace rapidjson;

  const std::string key("soname");
  if (!top.HasMember(key.c_str()))
  {
    fprintf(stderr, "%s:\n  Key '%s' not found in Document %s\n",
            __PRETTY_FUNCTION__, key.c_str(), detName.c_str());
    return nullptr;
  }
  std::string so(top[key.c_str()].GetString());
  if (so.length() == 0)  return nullptr;
  printf("Loading symbols from library '%s'\n", so.c_str());

  if (_object)                          // If the object exists,
    delete _object;                     // delete it before unloading the lib

  // Lib must remain open during Unconfig transition
  _dl.close();                          // If a lib is open, close it first

  if (_dl.open(so, RTLD_LAZY))
  {
    fprintf(stderr, "%s:\n  Could not open library '%s'\n",
            __PRETTY_FUNCTION__, so.c_str());
    return nullptr;
  }

  Create_t* createFn = reinterpret_cast<Create_t*>(_dl.loadSymbol(symbol.c_str()));
  if (!createFn)
  {
    fprintf(stderr, "%s:\n  Symbol '%s' not found in %s\n",
            __PRETTY_FUNCTION__, symbol.c_str(), so.c_str());
    return nullptr;
  }
  _object = createFn();
  if (!_object)
  {
    fprintf(stderr, "%s:\n  Error calling %s\n",
            __PRETTY_FUNCTION__, symbol.c_str());
    return nullptr;
  }
  return _object;
}

#endif
