#ifndef Pds_HSD_Server_hh
#define Pds_HSD_Server_hh

#include "psdaq/hsd/pds/utility/EbServer.hh"
#include "psdaq/hsd/pds/utility/BldSequenceSrv.hh"

#include <vector>

namespace Pds {
  namespace PvDaq {
    class Server : public EbServer,
		   public BldSequenceSrv {
    public:
      Server(const DetInfo&, int fd);
      ~Server();
    public:
      //  Eb-key interface
      EbServerDeclare;

      // Server interface
      int         fetch   (char*,int);
      void        dump    (int) const;
      bool        isValued() const;
      const Src&  client  () const;
      const Xtc&  xtc     () const;
      bool        more    () const;
      unsigned    length  () const;
      unsigned    offset  () const;
      // BldSequenceSrv
      unsigned    fiducials() const;
    private:
      Xtc      _xtc;
      unsigned _fid;
      int      _fd;
      char*    _evBuffer;
    };
  };
};

#endif
