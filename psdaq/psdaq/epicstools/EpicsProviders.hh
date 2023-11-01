#ifndef Pds_EpicsProviders_hh
#define Pds_EpicsProviders_hh

#include "pva/client.h"

namespace Pds_Epics {
    class EpicsProviders {
    private:
        EpicsProviders();
        ~EpicsProviders();
    public:
        static pvac::ClientProvider& pva();
        static pvac::ClientProvider& ca ();
    private:
        pvac::ClientProvider* _pva;
        pvac::ClientProvider* _ca;
    };
};

#endif
