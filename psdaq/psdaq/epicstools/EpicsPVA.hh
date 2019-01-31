#ifndef Pds_EpicsPVA_hh
#define Pds_EpicsPVA_hh

#include <iostream>
#include <chrono>
#include <future>
#include <thread>

#include "pva/client.h"
#include "pv/ntscalar.h"
#include "pv/pvIntrospect.h"
#include "pv/pvData.h"

#include "psdaq/epicstools/PVMonitorCb.hh"



namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;
namespace nt  = epics::nt;

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;
namespace nt  = epics::nt;


namespace Pds_Epics {
  class EpicsPVA :public pvac::ClientChannel::ConnectCallback, pvac::ClientChannel::GetCallback, pvac::ClientChannel::MonitorCallback {
  public:
    EpicsPVA(const char *channelName, const int maxElements=0);
    EpicsPVA(const char *channelName, PVMonitorCb*, const int maxElements=0);
    virtual ~EpicsPVA();

    std::string name() const { return _channel.name(); }
    bool connected() const { return _connected; }
    long sec();
    int nsec();
    size_t nelem();

    // Get the PV's value as the specified raw type using EPICS's comversion functions.
    // For example, uint32 val = _pv->getScalarAs<pvUInt>();
    template<typename T> T getScalarAs() const {
        if(_strct != NULL) return _strct->getSubField<pvd::PVScalar>("value")->getAs<T>();
        return 0;
    }

    template<typename T> void getVectorAs(pvd::shared_vector<const T> &vec) const {
        if(_strct != NULL) _strct->getSubField<pvd::PVScalarArray>("value")->getAs<T>(vec);
    }
    // This is not an efficient method; if the types match we should not do any copying.
    // However; this is how many, many macros are written; so this is a convienience method.
    // For a more efficient potentially zero copy call; use getVectorAs.
    template<typename T> T getVectorElemAt(size_t i) const {
        if(_strct == NULL) return 0;
        pvd::shared_vector<const T> vec;
        _strct->getSubField<pvd::PVScalarArray>("value")->getAs<T>(vec);
        return vec[i];
    }

    template<typename T> void putFrom(T val) {
        try {
            _channel.put().set("value", val).exec();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting to pv " << name() << std::endl;
        } catch(std::runtime_error r) {
            std::cout << "Runtime error when putting to pv " << name() << std::endl;
        }
    }

    template<typename T> void putFromVector(const pvd::shared_vector<const T>& val) {
        try {
            _channel.put().set("value", val).exec();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting a vector of size " << val.size() << " to pv " << name() << std::endl;
        }
    }

    virtual void getDone (const pvac::GetEvent &evt);
    virtual void connectEvent (const pvac::ConnectEvent &evt);
    virtual void monitorEvent (const pvac::MonitorEvent &evt);

  protected:
    virtual void onConnect(); // Propogate connection callback to subclasses.

    pvac::ClientChannel   _channel;
    pvac::Operation _op;
    pvac::Monitor _pvmon;
    std::promise<pvd::PVStructure::const_shared_pointer> _promise;
    pvd::PVStructure::const_shared_pointer _strct;

    PVMonitorCb*     _monitorCB;
    bool  _connected;

    bool getComplete();
    };
};

#endif
