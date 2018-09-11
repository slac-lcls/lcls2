#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfLink.hh"

namespace Pds {
  namespace Eb {

    class EbLfClient
    {
    public:
      EbLfClient();
      ~EbLfClient();
    public:
      int connect(const char* peer,
                  const char* port,
                  unsigned    tmo,
                  EbLfLink**  link);
    public:
      int shutdown(EbLfLink*);
    };
  };
};

#endif
