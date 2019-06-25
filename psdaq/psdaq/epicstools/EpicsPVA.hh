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
#include "pv/createRequest.h"

#include "psdaq/epicstools/PVMonitorCb.hh"

//static bool lfirst = true;

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;
namespace nt  = epics::nt;

namespace Pds_Epics {
    // Both the PutTracker's are copied over from the V4 example code.
    template<typename T> struct PutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(PutTracker);
        pvac::Operation op;
        const T value;
        PutTracker(pvac::ClientChannel& channel, const pvd::PVStructure::const_shared_pointer& pvReq, const T& val)
            :op(channel.put(this, pvReq)) ,value(val) {

            }

        virtual ~PutTracker() { op.cancel(); }

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
                std::cerr<<op.name()<<" Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<op.name()<<" Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                // std::cout<<op.name()<<" Done\n";
                break;
            }

            delete this;
        }
    };

    template<typename T> struct VectorPutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(VectorPutTracker);
        pvac::Operation op;
        const pvd::shared_vector<const T> value;
        VectorPutTracker(pvac::ClientChannel& channel, const pvd::PVStructure::const_shared_pointer& pvReq, const pvd::shared_vector<const T>& val)
            :op(channel.put(this, pvReq)) ,value(val) {

            }

        virtual ~VectorPutTracker() { op.cancel(); }

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
                std::cerr<<op.name()<<" Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<op.name()<<" Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                // std::cout<<op.name()<<" Done\n";
                break;
            }

            delete this;
        }
    };

    struct StructurePutTracker : public pvac::ClientChannel::PutCallback {
        POINTER_DEFINITIONS(StructurePutTracker);
        pvac::Operation op;
        const char* value;
        const unsigned* sizes;
        bool ldebug;
      StructurePutTracker(pvac::ClientChannel& channel, const pvd::PVStructure::const_shared_pointer& pvReq, const char* val, const unsigned* sz, bool debug)
        :op(channel.put(this, pvReq)) ,value(val), sizes(sz), ldebug(debug) {
        }

        virtual ~StructurePutTracker() { op.cancel(); }

        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args);
        virtual void putDone(const pvac::PutEvent &evt) OVERRIDE FINAL
        {
            switch(evt.event) {
            case pvac::PutEvent::Fail:
                std::cerr<<op.name()<<" putDone Error: "<<evt.message<<"\n";
                break;
            case pvac::PutEvent::Cancel:
                std::cerr<<op.name()<<" Cancelled\n";
                break;
            case pvac::PutEvent::Success:
                // std::cout<<op.name()<<" Done\n";
                break;
            }

            delete this;
        }
    };

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
            new PutTracker<T>(_channel, pvd::CreateRequest::create()->createRequest("field()"), val);
            // _channel.put().set("value", val).exec();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting to pv " << name() << std::endl;
        } catch(std::runtime_error r) {
            std::cout << "Runtime error when putting to pv " << name() << std::endl;
        }
    }

    template<typename T> void putFromVector(const pvd::shared_vector<const T>& val) {
        try {
            new VectorPutTracker<T>(_channel, pvd::CreateRequest::create()->createRequest("field()"), val);
            // _channel.put().set("value", val).exec();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting a vector of size " << val.size() << " to pv " << name() << std::endl;
        }
    }

    void putFromStructure(const void* val, const unsigned* sizes, bool ldebug=false) {
      try {
        new StructurePutTracker(_channel, pvd::CreateRequest::create()->createRequest("field()"), reinterpret_cast<const char*>(val), sizes, ldebug);
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
    bool  _connected;

    bool getComplete();
  };
};

#endif
