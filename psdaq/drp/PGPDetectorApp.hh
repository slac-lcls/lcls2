/**
 * @file
 * @brief PGPDetectorApp, the collection application of the PGP-card DRP (drp.cc).
 */
#pragma once

#include <thread>
#include <Python.h>
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "psdaq/service/Collection.hh"

namespace Drp {

/** CollectionApp of the PGP-card DRP: creates the detector chosen by detType and the PGPDrp, optionally starts the Python DRP worker processes, and handles the collection transitions. */
class PGPDetectorApp : public CollectionApp
{
public:
    /** Register with the collection as a drp, open the PGP pool, initialize Python and release the GIL. */
    PGPDetectorApp(Parameters& para);
    /** Call handleReset(), clean up the Python DRP message queues and shared memory if used, free the ID arrays and the detector, and finalize Python. */
    virtual ~PGPDetectorApp();
    /** Create the detector registered for detType (fakecam, cspad, hsd, epixquad, epixquad1kfps, epixhr2x2, epixhremu, epixm320, epixUHR, epix100, jungfrau, jungfrauemu, opal, tt, tb, ts, wave8, wave8he, hrencoder, piranha4, epixuhr3x2) and the PGPDrp, and set up the Python DRP when the drp kwarg is python. Throws if the detector type is unknown or the Python DRP setup fails. */
    void initialize();
    /** Return JSON connect_info with the NIC IP address (from the ep_domain kwarg, or the forceEnet kwarg), merged with the detector connection info and DrpBase::connectionInfo(). */
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
    /** Call the detector connectionShutdown() and DrpBase::shutdown(). */
    void connectionShutdown() override;
    /** Unsubscribe from the partition, unconfigure, disconnect and shut down the connection; with the Python DRP, also drain its message queues and (unless msg is empty, as on exit) reset it. */
    void handleReset(const nlohmann::json& msg) override;
private:
    void handleDealloc(const nlohmann::json& msg) override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void handlePhase1(const nlohmann::json& msg) override;
    int setupDrpPython();
    std::string _disable(XtcData::Xtc& xtc, const void* const bufEnd,
                         const nlohmann::json& phase1Info);
    std::string _endrun(const nlohmann::json& phase1Info);
    void _unconfigure();
    void _disconnect();
    void drainDrpMessageQueues();
    int resetDrpPython();
    Parameters& m_para;
    MemPoolCpu m_pool;
    Detector* m_det;
    std::unique_ptr<PGPDrp> m_drp;
    bool m_unconfigure;
    std::string m_lastKey;
    PyThreadState* m_pysave;
    int* m_inpMqId;
    int* m_resMqId;
    int* m_inpShmId;
    int* m_resShmId;
    std::string keyBase;
    pid_t* m_drpPids;
    size_t m_shmemSize;
    bool m_pythonDrp;
};

}
