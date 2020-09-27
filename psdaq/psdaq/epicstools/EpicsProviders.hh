#ifndef Pds_EpicsProviders_hh
#define Pds_EpicsProviders_hh

#include "pva/client.h"

namespace Pds_Epics {
    class EpicsProviders {
    public:
        EpicsProviders();
        static pvac::ClientProvider& pva();
        static pvac::ClientProvider& ca ();
    };
};

#endif
