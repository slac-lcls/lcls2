/**
 * @file
 * @brief Declares XtcData::TransitionId, the enum of transition ids and its name lookup.
 */
#ifndef XtcData_TransitionId_hh
#define XtcData_TransitionId_hh

namespace XtcData
{

/** Wrapper class for the TransitionId::Value enum and its name() lookup. */
class TransitionId
{
public:
    /** Transition ids (0..12). Transition::service() decodes one of these from bits 27:24 of the datagram env word. */
    enum Value {
        // Must keep in synch with strings in src/TransitionId.cc
        // There is also math on these transition id numbers
        // in XtcMonitorServer.cc::_update that assumes they come in pairs.
        // the ConfigUpdate currently breaks this assumption.
        // there is also code in TransitionCache::allocate that
        // does math on transition id's.
        ClearReadout,  ///< Transition id 0.
        Reset,  ///< Transition id 1.
        Configure,  ///< Transition id 2.
        Unconfigure,  ///< Transition id 3.
        BeginRun,  ///< Transition id 4.
        EndRun,  ///< Transition id 5.
        BeginStep,  ///< Transition id 6.
        EndStep,  ///< Transition id 7.
        Enable,  ///< Transition id 8.
        Disable,  ///< Transition id 9.
        SlowUpdate,  ///< Transition id 10.
        Unused_11,  ///< Transition id 11; a placeholder name with no special handling in this header.
        L1Accept = 12, /**< Transition id 12; Transition::isEvent() is true for it. */       // Must be 12 to agree with firmware
        NumberOf  ///< Number of ids (13); name() returns "-Invalid-" for ids at or above it.
    };
    /** Return the enumerator name of id as a string (e.g. "Configure"), or "-Invalid-" if id >= NumberOf. */
    static const char* name(TransitionId::Value id);
};
}

#endif
