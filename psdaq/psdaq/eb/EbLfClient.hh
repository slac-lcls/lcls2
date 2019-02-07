#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfLink.hh"

namespace Pds {
  namespace Eb {

    class EbLfClient
    {
    public:
      EbLfClient(unsigned verbose);
    public:
      int connect(const char* peer,
                  const char* port,
                  unsigned    tmo,
                  EbLfLink**  link);
    public:
      const uint64_t& pending() const  { return _pending; }
    public:
      int shutdown(EbLfLink*);
    private:
      uint64_t _pending;                // Bit list of IDs currently posting
      unsigned _verbose;                // Print some stuff if set
    };
  };
};

#endif
