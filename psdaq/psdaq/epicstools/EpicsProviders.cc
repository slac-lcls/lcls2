#include "EpicsProviders.hh"

#include <pv/configuration.h>
#include <pv/caProvider.h>

namespace pva = epics::pvAccess;

namespace Pds_Epics {

    static EpicsProviders* _providers = nullptr;

    EpicsProviders::EpicsProviders() :
      _pva(nullptr),
      _ca (nullptr)
    {
        if (!_pva)
        {
            pva::Configuration::shared_pointer
              configuration(pva::ConfigurationBuilder()
                            .push_env()
                            .build());
            pva::ca::CAClientFactory::start();
            _ca  = new pvac::ClientProvider("ca" , configuration);
            _pva = new pvac::ClientProvider("pva", configuration);
        }
    }

    EpicsProviders::~EpicsProviders()
    {
        if (_providers)
        {
          if (_pva)  delete _pva;
          if (_ca)   delete _ca;
          delete _providers;
          _providers = nullptr;
        }
    }

    pvac::ClientProvider& EpicsProviders::pva()
    {
        if (_providers && _providers->_pva)  return *_providers->_pva;

        if (!_providers)  _providers = new EpicsProviders;

        return *_providers->_pva;
    }

    pvac::ClientProvider& EpicsProviders::ca()
    {
        if (_providers && _providers->_ca)  return *_providers->_ca;

        if (!_providers)  _providers = new EpicsProviders;

        return *_providers->_ca;
    }
};
