/**
 * @file
 * @brief Trigger helpers: fetchDocument() to read a configuration from the config database, and Factory to load a plugin from a shared library.
 */
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

    /** Call get_config_json of the Python module psdaq.configdb.get_config with connectMsg, configAlias and section (and 0), and parse the returned JSON into top. Returns 0, or -1 on a Python or parse error. Python must already be initialized (per the code comment). */
    int fetchDocument(const std::string&   connectMsg,
                      const std::string&   configAlias,
                      const std::string&   section,
                      rapidjson::Document& top);

    template <typename T>
    /** Loads a shared library and creates one object of type T with a C function exported by it; the object is deleted when the factory is destroyed or reused. */
    class Factory
    {
    public:
      /** Construct with no object. */
      Factory() : _object(nullptr) {}
      /** Delete the object if one was created. */
      ~Factory() { if (_object)  delete _object; }
    public:
      /** Delete any previous object, close any open library, open soname and call its function symbol to create a new object. Returns the object, or nullptr if the library, the symbol or the creation fails. */
      T* create(const std::string& soname,
                const std::string& symbol);
    private:
      typedef T* Create_t();
      Pds::Dl _dl;
      T*      _object;
    };
  };
};


template <typename T>
T* Pds::Trg::Factory<T>::create(const std::string& soname,
                                const std::string& symbol)
{
  using namespace rapidjson;
  using logging = psalg::SysLog;

  logging::debug("Loading library '%s'", soname.c_str());

  if (_object)                          // If the object exists,
  {
    delete _object;                     // delete it before unloading the lib
    _object = nullptr;
  }

  // Lib must remain open during Unconfig transition
  _dl.close();                          // If a lib is open, close it first

  if (_dl.open(soname, RTLD_LAZY))
  {
    logging::debug("Error opening library '%s' for symbol '%s'",
                   soname.c_str(), symbol.c_str());
    return nullptr;
  }

  Create_t* createFn = reinterpret_cast<Create_t*>(_dl.loadSymbol(symbol.c_str()));
  if (!createFn)
  {
    logging::debug("Symbol '%s' not found in %s", symbol.c_str(), soname.c_str());
    return nullptr;
  }
  _object = createFn();
  if (!_object)
  {
    logging::debug("Error calling %s", symbol.c_str());
    return nullptr;
  }
  logging::debug("Loaded library '%s'", soname.c_str());
  return _object;
}

#endif
