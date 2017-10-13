#ifndef Pds_Eb_XptInlet_hh
#define Pds_Eb_XptInlet_hh

namespace Pds {
  namespace Eb {

    class XptInlet
    {
    public:
      XptInlet(std::string& port);
      virtual ~XptInlet();
    public:
      virtual int  connect() = 0;
      virtual void pend()    = 0;
    private:
      std::string& _port
    };
  };
};

#endif
