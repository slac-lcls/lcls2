#ifndef Pds_Eb_EbLfClient_hh
#define Pds_Eb_EbLfClient_hh

#include "EbLfLink.hh"

namespace Pds {
  namespace Eb {

    class EbLfClient
    {
    public:
      EbLfClient(unsigned verbose);
      ~EbLfClient();
    public:
      int connect(const char* peer,
                  const char* port,
                  unsigned    tmo,
                  EbLfLink**  link);
    public:
      const uint64_t& pending() const;
    public:
      int shutdown(EbLfLink*);
    private:
      unsigned _verbose;
    private:
      uint64_t _pending;
    };
  };
};

inline
const uint64_t& Pds::Eb::EbLfClient::pending() const
{
  return _pending;
}

#endif
