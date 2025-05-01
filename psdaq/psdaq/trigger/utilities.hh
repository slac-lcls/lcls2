#ifndef Pds_Trg_utilities_hh
#define Pds_Trg_utilities_hh

#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>
#include "psalg/utils/SysLog.hh"

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
      ~Factory() { if (_object)  delete _object; }
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
                                const std::string&         docName,
                                const std::string&         symbol)
{
  using namespace rapidjson;
  using logging = psalg::SysLog;

  const std::string key("soname");
  if (!top.HasMember(key.c_str()))
  {
    logging::debug("Key '%s' not found in Document %s",
                   key.c_str(), docName.c_str());
    return nullptr;
  }
  std::string so(top[key.c_str()].GetString());
  if (so.length() == 0)
  {
    logging::debug("Empty library name for key '%s'", key.c_str());
    return nullptr;
  }
  logging::debug("Loading symbols from library '%s'", so.c_str());

  if (_object)                          // If the object exists,
  {
    delete _object;                     // delete it before unloading the lib
    _object = nullptr;
  }

  // Lib must remain open during Unconfig transition
  _dl.close();                          // If a lib is open, close it first

  if (_dl.open(so, RTLD_LAZY))
  {
    logging::debug("Error opening library '%s' for symbol '%s'",
                   so.c_str(), symbol.c_str());
    return nullptr;
  }

  Create_t* createFn = reinterpret_cast<Create_t*>(_dl.loadSymbol(symbol.c_str()));
  if (!createFn)
  {
    logging::debug("Symbol '%s' not found in %s", symbol.c_str(), so.c_str());
    return nullptr;
  }
  _object = createFn();
  if (!_object)
  {
    logging::debug("Error calling %s", symbol.c_str());
    return nullptr;
  }
  logging::debug("Loaded library '%s'", so.c_str());
  return _object;
}

#endif
