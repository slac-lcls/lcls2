#include "EpicsProviders.hh"

#include <pv/configuration.h>
#include <pv/caProvider.h>

namespace pva = epics::pvAccess;

namespace Pds_Epics {

    static pvac::ClientProvider* _pva;
    static pvac::ClientProvider* _ca;

    pvac::ClientProvider& EpicsProviders::pva() { return *_pva; }
    pvac::ClientProvider& EpicsProviders::ca () { return *_ca ; }

    EpicsProviders::EpicsProviders()
    {
        pva::Configuration::shared_pointer
          configuration(pva::ConfigurationBuilder()
                        .push_env()
                        .build());
        pva::ca::CAClientFactory::start();
        _ca  = new pvac::ClientProvider("ca" , configuration);
        _pva = new pvac::ClientProvider("pva", configuration);
    }

    static EpicsProviders _providers;
};
