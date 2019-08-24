#include "Trigger.hh"

#include "psdaq/eb/utilities.hh"

#include "rapidjson/document.h"

using namespace Pds;
using namespace Pds::Eb;


Trigger* Pds::Eb::TriggerFactory::load(const std::string& connectMsg,
                                       const std::string& configAlias,
                                       const std::string& detName)
{
  using namespace rapidjson;

  Document top;
  int      rc = fetchDocument(connectMsg, configAlias, detName, top);
  if (rc)
  {
    fprintf(stderr, "%s:\n  Failed to find Document '%s' in ConfigDb\n",
            __PRETTY_FUNCTION__, detName.c_str());
    return nullptr;
  }
  const char* key("soname");
  if (!top.HasMember(key))
  {
    fprintf(stderr, "%s:\n  Key '%s' not found in Document %s\n",
            __PRETTY_FUNCTION__, key, detName.c_str());
    return nullptr;
  }
  std::string so(top[key].GetString());
  printf("Loading object symbols from library '%s'\n", so.c_str());

  if (_object)                          // If the object exists,
    delete _object;                     // delete it before unloading the lib

  // Lib must remain open during Unconfig transition
  _dl.close();                      // If a lib is open, close it first

  rc = _dl.open(so, RTLD_LAZY);
  if (rc)
  {
    fprintf(stderr, "%s:\n  Could not open library '%s'\n",
            __PRETTY_FUNCTION__, so.c_str());
    return nullptr;
  }

  Create_t* createFn = reinterpret_cast<Create_t*> (_dl.loadSymbol("create"));
  if (!createFn)
  {
    fprintf(stderr, "%s:\n  Decide object's create() (%p) not found in %s\n",
            __PRETTY_FUNCTION__, createFn, so.c_str());
    return nullptr;
  }
  _object = createFn();
  if (!_object)
  {
    fprintf(stderr, "%s:\n  Failed to create object\n",
            __PRETTY_FUNCTION__);
    return nullptr;
  }
  return _object;
}
