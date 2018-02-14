#ifndef Psdaq_App_Utils_hh
#define Psdaq_App_Utils_hh

namespace Psdaq {
  class AppUtils {
  public:
    static unsigned parse_ip(const char* ipString);
    static unsigned parse_interface(const char* interfaceString);
  };
};

#endif
