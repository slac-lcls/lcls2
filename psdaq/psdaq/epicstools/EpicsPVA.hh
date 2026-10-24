/**
 * @file
 * @brief EpicsPVA, the psdaq client of one EPICS PV over pvAccess or Channel Access, and the put callbacks it uses.
 */
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
    /** Put callback that writes one scalar to the value field of a PV and signals completionEvent when the put ends (copied from the EPICS V4 example code, per the code comment). */
    template<typename T> struct PutTracker : public pvac::ClientChannel::PutCallback {
        /** Expands the pvData POINTER_DEFINITIONS macro for this type; doxygen lists it as a function. */
        POINTER_DEFINITIONS(PutTracker);
        epicsEvent completionEvent;  ///< Signalled by putDone().
        const T value;  ///< Value to put.
        /** Keep a copy of val. */
        PutTracker(const T& val) : value(val) {}

        /** Does nothing; empty virtual destructor. */
        virtual ~PutTracker() {}

        /** Create the put structure from build, set its value field from value and mark only that field to be sent. */
        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args) {
            pvd::PVStructurePtr root(pvd::getPVDataCreate()->createPVStructure(build));
            pvd::PVScalarPtr valfld(root->getSubFieldT<pvd::PVScalar>("value"));
            valfld->putFrom(value);
            args.root = root;
            args.tosend.set(valfld->getFieldOffset());
            //            std::cerr << "Putting to PV " << op.name() << " " << valfld << std::endl;
        }
        /** Print a message for a failed or cancelled put, then signal completionEvent. */
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

    /** Put callback that writes an array to the value field of a PV and signals completionEvent when the put ends. */
    template<typename T> struct VectorPutTracker : public pvac::ClientChannel::PutCallback {
        /** Expands the pvData POINTER_DEFINITIONS macro for this type; doxygen lists it as a function. */
        POINTER_DEFINITIONS(VectorPutTracker);
        epicsEvent completionEvent;  ///< Signalled by putDone().
        const pvd::shared_vector<const T> value;  ///< Array to put.
        /** Keep a shared copy of val. */
        VectorPutTracker(const pvd::shared_vector<const T>& val) : value(val) {}

        /** Does nothing; empty virtual destructor. */
        virtual ~VectorPutTracker() {}

        /** Create the put structure from build, set its value array from value and mark only that field to be sent. */
        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args) {
            pvd::PVStructurePtr root(pvd::getPVDataCreate()->createPVStructure(build));
            pvd::PVScalarArrayPtr valfld(root->getSubFieldT<pvd::PVScalarArray>("value"));
            valfld->putFrom(value);
            args.root = root;
            args.tosend.set(valfld->getFieldOffset());
            //            std::cerr << "Putting to PV " << op.name() << " " << valfld << std::endl;
        }
        /** Print a message for a failed or cancelled put, then signal completionEvent. */
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

    /** Put callback that fills the top-level fields of a PV structure from a packed buffer and signals completionEvent when the put ends. */
    struct StructurePutTracker : public pvac::ClientChannel::PutCallback {
        /** Expands the pvData POINTER_DEFINITIONS macro for this type; doxygen lists it as a function. */
        POINTER_DEFINITIONS(StructurePutTracker);
        epicsEvent completionEvent;  ///< Signalled by putDone().
        const char* value;  ///< Packed field values, read in field order; putBuild() advances this pointer.
        const unsigned* sizes;  ///< Element counts of the array fields, in field order; putBuild() advances this pointer.
        bool ldebug;  ///< If true, putBuild() dumps the structure and prints each value read.
        /** Store the buffer and size pointers (not copied) and the debug flag. */
        StructurePutTracker(const char* val, const unsigned* sz, bool debug)
          : value(val), sizes(sz), ldebug(debug) {
        }

        /** Does nothing; empty virtual destructor. */
        virtual ~StructurePutTracker() {}

        /** Create the put structure from build and fill its top-level scalar and scalar array fields in order from value, as int32, int64, uint32, uint64, float or double (array lengths from sizes). Other field kinds are skipped and other element types throw a std::string; bit 0 of the send mask is set. */
        virtual void putBuild(const epics::pvData::StructureConstPtr &build, pvac::ClientChannel::PutCallback::Args& args);
        /** Print a message for a failed or cancelled put, then signal completionEvent. */
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

  /** Client of one EPICS PV over pvAccess or Channel Access. It connects asynchronously, gets the value once connected and, if a PVMonitorCb is given, then monitors the PV and calls its updated() for each update. getScalarAs() and related methods read the latest value; the put methods block until the put ends. */
  class EpicsPVA :public pvac::ClientChannel::ConnectCallback, pvac::ClientChannel::GetCallback, pvac::ClientChannel::MonitorCallback {
  public:
    /** Connect to channelName through pva without a monitor callback; maxElements is not used. */
    EpicsPVA(const char *channelName, const int maxElements=0);
    /** Connect to channelName through pva; monitor, if not null, receives updated() calls. maxElements is not used. */
    EpicsPVA(const char *channelName, PVMonitorCb*, const int maxElements=0);
    /** Connect to channelName through ca if provider is ca, otherwise pva. If nType is true, monitoring requests only the timeStamp and value fields; maxElements is not used. */
    EpicsPVA(const char* provider, const char *channelName, PVMonitorCb*, const int maxElements=0, bool nType=false);
    /** Remove the connect listener and cancel the pending get. */
    virtual ~EpicsPVA();

    /** Declared but not defined in EpicsPVA.cc. */
    static void setProvider(const char*);

    /** Return the channel name. */
    std::string name() const { return _channel.name(); }
    /** Return the state from the last connect event; the flag is not initialized before the first event. */
    bool connected() const { return _connected; }
    /** Return secondsPastEpoch of the PV time stamp, or -1 if no value arrives within 30 s (getComplete()). */
    long sec();
    /** Return the nanoseconds of the PV time stamp, or -1 if no value arrives within 30 s. */
    int nsec();
    /** Return the length of the value array, or the largest size_t value (from -1) if no value arrives within 30 s. */
    size_t nelem();

    // Get the PV's value as the specified raw type using EPICS's comversion functions.
    // For example, uint32 val = _pv->getScalarAs<pvUInt>();
    /** Return field (default value) of the latest structure converted to T, or 0 if no value has been received. */
    template<typename T> T getScalarAs(const char* field = "value") const {
        if(_strct != NULL) return _strct->getSubField<pvd::PVScalar>(field)->getAs<T>();
        return 0;
    }

    /** Copy array field (default value) of the latest structure into vec as T; does nothing if no value has been received. */
    template<typename T> void getVectorAs(pvd::shared_vector<const T> &vec, const char* field="value") const {
        if(_strct != NULL) _strct->getSubField<pvd::PVScalarArray>(field)->getAs<T>(vec);
    }
    // This is not an efficient method; if the types match we should not do any copying.
    // However; this is how many, many macros are written; so this is a convienience method.
    // For a more efficient potentially zero copy call; use getVectorAs.
    /** Return element i of array field converted to T, converting the whole array first (per the code comment); i is not range-checked. Returns 0 if no value has been received. */
    template<typename T> T getVectorElemAt(size_t i, const char* field="value") const {
        if(_strct == NULL) return 0;
        pvd::shared_vector<const T> vec;
        _strct->getSubField<pvd::PVScalarArray>(field)->getAs<T>(vec);
        return vec[i];
    }

    /** Put val to the value field and wait for the put to end; timeouts and runtime errors are caught and printed. */
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

    /** Put array val to the value field and wait for the put to end; timeouts are caught and printed. */
    template<typename T> void putFromVector(const pvd::shared_vector<const T>& val) {
        try {
	  VectorPutTracker<T> putter(val);
	  pvac::Operation op = _channel.put(&putter,pvd::CreateRequest::create()->createRequest("field()"));
	  putter.completionEvent.wait();
        } catch(const pvac::Timeout& t) {
            std::cout << "Timeout when putting a vector of size " << val.size() << " to pv " << name() << std::endl;
        }
    }

    /** Put the packed buffer val to the top-level fields of the PV (see StructurePutTracker) and wait for the put to end; timeouts are caught and printed. */
    void putFromStructure(const void* val, const unsigned* sizes, bool ldebug=false) {
      try {
	StructurePutTracker putter(reinterpret_cast<const char*>(val), sizes, ldebug);
	pvac::Operation op = _channel.put(&putter,pvd::CreateRequest::create()->createRequest("field()"));
	putter.completionEvent.wait();
      } catch(const pvac::Timeout& t) {
        std::cout << "Timeout when putting a structure to pv " << name() << std::endl;
      }
    }

    /** Get callback: on success, fulfil the value promise and, if a monitor callback was given, start monitoring (only timeStamp and value if nType is set). Failures and cancellations are printed. */
    virtual void getDone (const pvac::GetEvent &evt);
    /** Connect callback: store the connection state and, if connected, start a get of the PV. */
    virtual void connectEvent (const pvac::ConnectEvent &evt);
    /** Monitor callback: for data, store each queued update as the latest structure and call the monitor callback's updated(). Failures, cancels and disconnects are only printed. */
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
