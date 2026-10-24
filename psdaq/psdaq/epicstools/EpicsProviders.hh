/**
 * @file
 * @brief EpicsProviders, the process-wide pvAccess and Channel Access client providers.
 */
#ifndef Pds_EpicsProviders_hh
#define Pds_EpicsProviders_hh

#include "pva/client.h"

/** Namespace of the psdaq EPICS client helpers (pvAccess and Channel Access). */
namespace Pds_Epics {
    /** Process-wide pva and ca client providers, both created on first use from the EPICS environment configuration. The lazy creation is not protected against concurrent first calls. */
    class EpicsProviders {
    private:
        EpicsProviders();
        ~EpicsProviders();
    public:
        /** Return the shared pva client provider, creating both providers on first use. */
        static pvac::ClientProvider& pva();
        /** Return the shared Channel Access client provider, creating both providers on first use (which also starts the CA client factory). */
        static pvac::ClientProvider& ca ();
    private:
        pvac::ClientProvider* _pva;
        pvac::ClientProvider* _ca;
    };
};

#endif
