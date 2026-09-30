#ifndef Pds_EpicsPVA_hh
#define Pds_EpicsPVA_hh

#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include <stdexcept>

#include "pva/client.h"
#include "pv/ntscalar.h"
#include "pv/pvIntrospect.h"
#include "pv/pvData.h"
#include "pv/createRequest.h"
#include <epicsEvent.h>

#include "psdaq/epicstools/PVMonitorCb.hh"

//static bool lfirst = true;

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;
namespace nt  = epics::nt;

namespace Pds_Epics {
    // Both the PutTracker's are copied over from the V4 example code.
    template<typename T> struct PutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(PutTracker);
        epicsEvent completionEvent;
        const T value;
        PutTracker(const T& val) : value(val) {}

        virtual ~PutTracker() {}

        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args) {
            pvd::PVStructurePtr root(pvd::getPVDataCreate()->createPVStructure(build));
            pvd::PVScalarPtr valfld(root->getSubFieldT<pvd::PVScalar>("value"));
            valfld->putFrom(value);
            args.root = root;
            args.tosend.set(valfld->getFieldOffset());
            //            std::cerr << "Putting to PV " << op.name() << " " << valfld << std::endl;
        }
        virtual void putDone(const pvac::PutEvent &evt) OVERRIDE FINAL
        {
            switch(evt.event) {
            case pvac::PutEvent::Fail:
                std::cerr<<"putDone Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<"putDone Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                break;
            }

	    completionEvent.signal();
        }
    };

    template<typename T> struct VectorPutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(VectorPutTracker);
        epicsEvent completionEvent;
        const pvd::shared_vector<const T> value;
        VectorPutTracker(const pvd::shared_vector<const T>& val) : value(val) {}

        virtual ~VectorPutTracker() {}

        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args) {
            pvd::PVStructurePtr root(pvd::getPVDataCreate()->createPVStructure(build));
            pvd::PVScalarArrayPtr valfld(root->getSubFieldT<pvd::PVScalarArray>("value"));
            valfld->putFrom(value);
            args.root = root;
            args.tosend.set(valfld->getFieldOffset());
            //            std::cerr << "Putting to PV " << op.name() << " " << valfld << std::endl;
        }
        virtual void putDone(const pvac::PutEvent &evt) OVERRIDE FINAL
        {
            switch(evt.event) {
            case pvac::PutEvent::Fail:
	      std::cerr<<"putDone Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<"putDone Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                // std::cout<<op.name()<<" Done\n";
                break;
            }

	    completionEvent.signal();
        }
    };

    struct StructurePutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(StructurePutTracker);
        epicsEvent completionEvent;
        const char* value;
        const unsigned* sizes;
        bool ldebug; 
        StructurePutTracker(const char* val, const unsigned* sz, bool debug)
          : value(val), sizes(sz), ldebug(debug) {
        }

        virtual ~StructurePutTracker() {}

        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args);
        virtual void putDone(const pvac::PutEvent &evt) OVERRIDE FINAL
        {
            switch(evt.event) {
            case pvac::PutEvent::Fail:
 	        std::cerr<<"putDone Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<"putDone Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                break;
            }

	    completionEvent.signal();
        }
    };

  class EpicsPVA :public pvac::ClientChannel::ConnectCallback, pvac::ClientChannel::GetCallback, pvac::ClientChannel::MonitorCallback {
  public:
    EpicsPVA(const char *channelName, const int maxElements=0);
    EpicsPVA(const char *channelName, PVMonitorCb*, const int maxElements=0);
    EpicsPVA(const char* provider, const char *channelName, PVMonitorCb*, const int maxElements=0, bool nType=false);
    virtual ~EpicsPVA();

    static void setProvider(const char*);

    std::string name() const { return _channel.name(); }
    bool connected() const { return _connected; }
    long sec();
    int nsec();
    size_t nelem();

    // Get the PV's value as the specified raw type using EPICS's comversion functions.
    // For example, uint32 val = _pv->getScalarAs<pvUInt>();
    template<typename T> T getScalarAs(const char* field = "value") const {
        if(_strct != NULL) return _strct->getSubField<pvd::PVScalar>(field)->getAs<T>();
        return 0;
    }

    template<typename T> void getVectorAs(pvd::shared_vector<const T> &vec, const char* field="value") const {
        if(_strct != NULL) _strct->getSubField<pvd::PVScalarArray>(field)->getAs<T>(vec);
    }
    // This is not an efficient method; if the types match we should not do any copying.
    // However; this is how many, many macros are written; so this is a convienience method.
    // For a more efficient potentially zero copy call; use getVectorAs.
    template<typename T> T getVectorElemAt(size_t i, const char* field="value") const {
        if(_strct == NULL) return 0;
        pvd::shared_vector<const T> vec;
        _strct->getSubField<pvd::PVScalarArray>(field)->getAs<T>(vec);
        return vec[i];
    }

    template<typename T> void putFrom(T val) {
        try {
	  PutTracker<T> putter(val);
	  pvac::Operation op = _channel.put(&putter,pvd::CreateRequest::create()->createRequest("field()"));
	  putter.completionEvent.wait();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting to pv " << name() << std::endl;
        } catch(const std::runtime_error& r) {
            std::cout << "Runtime error when putting to pv " << name() << std::endl;
        }
    }

    template<typename T> void putFromVector(const pvd::shared_vector<const T>& val) {
        try {
	  VectorPutTracker<T> putter(val);
	  pvac::Operation op = _channel.put(&putter,pvd::CreateRequest::create()->createRequest("field()"));
	  putter.completionEvent.wait();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting a vector of size " << val.size() << " to pv " << name() << std::endl;
        }
    }

    void putFromStructure(const void* val, const unsigned* sizes, bool ldebug=false) {
      try {
	StructurePutTracker putter(reinterpret_cast<const char*>(val), sizes, ldebug);
	pvac::Operation op = _channel.put(&putter,pvd::CreateRequest::create()->createRequest("field()"));
	putter.completionEvent.wait();
      } catch(const pvac::Timeout& t) {
        std::cout << "Timeout when putting a structure to pv " << name() << std::endl;
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
    bool  _nType;
    bool  _connected;

    bool getComplete(unsigned tmo);   // seconds
    bool getComplete() { return getComplete(30); }
  };
};

#endif
