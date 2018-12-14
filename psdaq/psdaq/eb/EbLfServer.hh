#ifndef Pds_Eb_EbLfServer_hh
#define Pds_Eb_EbLfServer_hh

#include "EbLfLink.hh"

#include <stdint.h>
#include <cstddef>


struct fi_cq_data_entry;

namespace Pds {

  namespace Fabrics {
    class PassiveEndpoint;
    class CompletionQueue;
  };

  namespace Eb {

    class EbLfServer
    {
    public:
      EbLfServer(const char* addr,
                 const char* port);
      ~EbLfServer();
    public:
      int connect(EbLfLink**, int msTmo = -1);
      int pend(fi_cq_data_entry*, int msTmo);
      int pend(void** context, int msTmo);
      int pend(uint64_t* data, int msTmo);
      int poll(uint64_t* data);
    public:
      const uint64_t& pending() const;
    public:
      int shutdown(EbLfLink*);
    private:
      int _initialize(const char* addr, const char* port);
      int _poll(fi_cq_data_entry*, uint64_t flags);
    private:
      int                       _status;
      Fabrics::PassiveEndpoint* _pep;  // Endpoint for establishing connections
      Fabrics::CompletionQueue* _rxcq;
      size_t                    _bufSize;
      int                       _tmo;
    private:
      uint64_t                  _pending;
      uint64_t                  _unused;
    };
  };
};


inline
const uint64_t& Pds::Eb::EbLfServer::pending() const
{
  return _pending;
}

#endif
