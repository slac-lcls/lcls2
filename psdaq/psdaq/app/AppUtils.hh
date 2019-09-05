#ifndef Psdaq_App_Utils_hh
#define Psdaq_App_Utils_hh

#include <vector>
#include <string>
#include <stdint.h>
#include <pthread.h>

namespace Psdaq {
  class MonitorArgs {
  public:
    void add(const char* title, const char* unit, uint64_t& value);
  public:
    std::vector<std::string> titles;
    std::vector<std::string> units;   // e.g. "Hz", "B/s"
    std::vector<uint64_t*  > values;
  };

  class AppUtils {
  public:
    static unsigned    parse_ip(const char* ipString);
    static unsigned    parse_interface(const char* interfaceString);
    static std::string parse_paddr(unsigned);
    static pthread_t   monitor(const MonitorArgs&);
  };
};

#endif
