#include "Trigger.hh"

#include "utilities.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "psdaq/service/IpcUtils.hh"
#include "xtcdata/xtc/TransitionId.hh"

#include <cstdint>
#include <chrono>
#include <vector>

#include <unistd.h>
#include <string>
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/msg.h>
#include <sys/wait.h>

using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;


namespace Pds {
  namespace Trg {

    class TebPyTrig : public Trigger
    {
    public:
      TebPyTrig();
      virtual ~TebPyTrig();
      unsigned rogReserve(unsigned rog,
                          unsigned meb,
                          size_t   nBufs) const;
      int  configure(const json&              connectMsg,
                     const json&              configureMsg,
                     const Pds::Eb::EbParams& prms) override;
      int  initialize(const std::vector<size_t>& inputsRegSize,
                      size_t                     resultsRegSize) override;
      void event(const Pds::EbDgram* const* start,
                 const Pds::EbDgram**       end,
                 Pds::Eb::ResultDgram&      result) override;
      void transition(Pds::Eb::ResultDgram& result) override;
      void shutdown() override;
      void cleanup();
    private:
      int  _startPython(pid_t& pyPid);
      int  _setupMsgQueue(std::string key, const char* name, int& id, bool write);
      int  _setupShMem(std::string key,
                       size_t      size,
                       const char* name,
                       int&        id,
                       void*&      data,
                       bool        write);
      int _checkPy(pid_t, bool wait = false);
      int _send(int mqId, const char*, size_t);
      int _recv(int mqId, char*, size_t, unsigned msTmo);
    private:
      std::string _connectMsg;
      std::string _pythonScript;
      unsigned    _rogRsrvdBuf[Pds::Eb::NUM_READOUT_GROUPS];
    private:
      std::string        _keyBase;
      unsigned           _partition;
      pid_t              _pyPid;
      int                _inpMqId;
      int                _resMqId;
      int                _inpShmId;
      int                _resShmId;
      std::vector<void*> _inpData;
      void*              _resData;
    };
  };
};


static Pds::Trg::TebPyTrig* _tebPyTrigger = nullptr;

static void cleanupOnSignal(int signum)
{
  logging::info("[C++] Cleaning up on receiving signal (%d)", signum);

  if (_tebPyTrigger)  _tebPyTrigger->shutdown();

  exit(signum);
}

Pds::Trg::TebPyTrig::TebPyTrig() :
  Trigger(),
  _pyPid   (0),
  _inpMqId (0),
  _resMqId (0),
  _inpShmId(0),
  _resShmId(0),
  _resData (nullptr)
{
  _tebPyTrigger = this;

  signal(SIGINT, cleanupOnSignal);

  logging::info("[C++] TebPyTrig has loaded");
}

Pds::Trg::TebPyTrig::~TebPyTrig()
{
  shutdown();
}

void Pds::Trg::TebPyTrig::shutdown()
{
  if (_tebPyTrigger)
  {
    logging::info("[C++] Stopping C++ side");

    if (_inpMqId)
    {
      int       rc;
      char msg[512];
      msg[0] = 's';
      rc = _send(_inpMqId, msg, 1);
      if (rc == 0)
        if (_pyPid)  _checkPy(_pyPid, true);
    }

    cleanup();

    _tebPyTrigger = nullptr;
  }
}

unsigned Pds::Trg::TebPyTrig::rogReserve(unsigned rog,
                                         unsigned meb,
                                         size_t   nBufs) const
{
  unsigned rsrvd = _rogRsrvdBuf[rog];
  return rsrvd < nBufs ? rsrvd : nBufs;
}

int Pds::Trg::TebPyTrig::configure(const json&              connectMsg,
                                   const json&              configureMsg,
                                   const Pds::Eb::EbParams& prms)
{
  int rc = 0;
  const json& top{configureMsg["trigger_body"]};

  _connectMsg = connectMsg.dump();

# define _FETCH(key, item)                                              \
  if (top.find(key) != top.end())  item = top[key];                     \
  else { fprintf(stderr, "%s:\n  Key '%s' not found\n",                 \
                 __PRETTY_FUNCTION__, key);  rc = -1; }

  _FETCH("pythonScript", _pythonScript);

  for (unsigned rog = 0; rog < Pds::Eb::NUM_READOUT_GROUPS; ++rog)
  {
    char key[40];
    snprintf(key, sizeof(key), "rogRsrvdBuf[%u]", rog);
    _FETCH(key, _rogRsrvdBuf[rog]);
  }

# undef _FETCH

  auto scriptPath = prms.kwargs.find("script_path") != prms.kwargs.end()
                  ? const_cast<Pds::Eb::EbParams&>(prms).kwargs["script_path"]
                  : ".";                // Revisit: Good default?
  _pythonScript = scriptPath + "/" + _pythonScript;
  _partition    = prms.partition;

  _keyBase = "p" + std::to_string(prms.partition) + "_teb" + std::to_string(prms.id);

  _inpMqId  = 0;
  _resMqId  = 0;
  _inpShmId = 0;
  _inpData  .clear();
  _resShmId = 0;
  _resData  = nullptr;

  return rc;
}

int Pds::Trg::TebPyTrig::_startPython(pid_t& pyPid)
{
  // Fork
  pyPid = fork();

  if (pyPid == pid_t(0))
  {
    logging::info("Running 'python -u %s'", _pythonScript.c_str());

    // Executing external code - exec returns only on error
    int rc = execlp("python",
                    "python",
                    "-u",
                    _pythonScript.c_str(),
                    ("-p " + std::to_string(_partition)).c_str(),
                    ("-b " + _keyBase).c_str(),
                    nullptr);
    logging::error("Error on exec 'python -u %s': %m", _pythonScript);
    return rc;
  }
  return 0;
}

int Pds::Trg::TebPyTrig::_setupMsgQueue(std::string key,
                                        const char* name,
                                        int&        mqId,
                                        bool        write)
{
  int rc = Pds::Ipc::setupDrpMsgQueue(key, 512, mqId, write);
  if ( rc == -1)
  {
    logging::error("[C++] Error in creating %s message queue with key %s: %m",
                   name, key.c_str());
    cleanup();
    return -1;
  }

  logging::info("[C++] %s message queues created", name);
  return 0;
}

int Pds::Trg::TebPyTrig::_setupShMem(std::string key,
                                     size_t      size,
                                     const char* name,
                                     int&        shmId,
                                     void*&      data,
                                     bool        write)
{
  int rc = Pds::Ipc::setupDrpShMem(key, size, shmId);
  if (rc == -1)
  {
    logging::error("[C++] Error in creating %s shared memory for key %s: %m",
                   name, key.c_str());
    cleanup();
    return -1;
  }

  rc = Pds::Ipc::attachDrpShMem(key, shmId, size, data, write);
  if (rc == -1)
  {
    logging::error("[C++] Error attaching %s shared memory for key %s: %m",
                   name, key.c_str());
    cleanup();
    return -1;
  }

  logging::info("[C++] %s shared memory created for key %s", name, key.c_str());

  return 0;
}

int Pds::Trg::TebPyTrig::_checkPy(pid_t pid, bool wait)
{
  pid_t child_status = waitpid(pid, NULL, wait ? 0 : WNOHANG);
  if (child_status != 0)
  {
    cleanup();
    return -1;
  }
  return 0;
}

int Pds::Trg::TebPyTrig::_send(int mqId, const char *msg, size_t size)
{

  int rc = Pds::Ipc::drpSend(mqId, msg, size);
  if (rc == -1)
  {
    logging::error("[C++] Error sending message '%c': %m",
                   msg[0]);
    cleanup();
    return -1;
  }
  return 0;
}

int Pds::Trg::TebPyTrig::_recv(int mqId, char *msg, size_t msgsize, unsigned msTmo)
{
  int rc = Pds::Ipc::drpRecv(mqId, msg, msgsize, msTmo);
  if (rc == -1)
  {
    logging::error("[C++] Error receiving message: %m");
    cleanup();
    return -1;
  }
  return 0;
}

void Pds::Trg::TebPyTrig::cleanup()
{
  Pds::Ipc::cleanupDrpMq("/mqtebinp_" + _keyBase, _inpMqId);
  Pds::Ipc::cleanupDrpMq("/mqtebqres_" + _keyBase, _resMqId);
  Pds::Ipc::cleanupDrpShmMem("/shmtebinp_" + _keyBase, _inpShmId);
  Pds::Ipc::cleanupDrpShmMem("/shmtebres_" + _keyBase, _resShmId);
}

int Pds::Trg::TebPyTrig::initialize(const std::vector<size_t>& inputsSizes,
                                    size_t                     resultsSize)
{
  int rc = 0;

  // Fork
  pid_t pyPid;
  rc = _startPython(pyPid);
  if (rc)  return rc;
  if (pyPid == pid_t(0))  return rc;
  _pyPid = pyPid;

  logging::info("[C++] Starting C++ side");

  // Creating message queues
  logging::info("[C++] Creating message queues");

  // Temporary solution to start from clean msg queues and shared memory
  std::remove(("/dev/mqueue/mqtebinp_" + _keyBase).c_str());
  std::remove(("/dev/mqueue/mqtebres_" + _keyBase).c_str());
  std::remove(("/dev/shm/shmtehinp_" + _keyBase).c_str());
  std::remove(("/dev/shm/shmtebres_" + _keyBase).c_str());

  rc = _setupMsgQueue("/mqtebinp_" + _keyBase, "Inputs", _inpMqId, true);
  if (rc)  return rc;

  rc = _setupMsgQueue("/mqtebres_" + _keyBase, "Results", _resMqId, false);
  if (rc)  return rc;

  // Creating shared memory
  logging::info("[C++] Creating shared memory blocks");

  // Calculate the size of the Inputs data block
  size_t inputsSize = 0;
  for (unsigned i = 0; i < inputsSizes.size(); ++i)
  {
    inputsSize += inputsSizes[i];
  }

  // Round up to an integral number of pages
  auto pageSize = sysconf(_SC_PAGESIZE);
  inputsSize = (inputsSize + pageSize - 1) & ~(pageSize - 1);

  void* inpData;
  rc = _setupShMem("/shmtebinp_" + _keyBase, inputsSize, "Inputs", _inpShmId, inpData, true);
  if (rc)  return rc;

  // Split up the Inputs data block into a buffer for each contributor
  // These buffers are in source ID order
  _inpData.resize(inputsSizes.size());
  for (unsigned i = 0; i < inputsSizes.size(); ++i)
  {
    _inpData[i] = inpData;
    inpData     = (char*)inpData + inputsSizes[i];
  }

  // Round up to an integral number of pages
  resultsSize = (resultsSize + pageSize - 1) & ~(pageSize - 1);

  rc = _setupShMem("/shmtebres_" + _keyBase, resultsSize, "Results", _resShmId, _resData, true);
  if (rc)  return rc;

  // Provide Inputs shared memory info to python
  logging::info("[C++] Sending Inputs shared memory info to Python");

  char msg[512];
  char* mtext = &msg[0];
  size_t size = sizeof(msg);
  msg[0] = 'i';
  int cnt = 1;
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%s", ("/shmtebinp_" + _keyBase).c_str());
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%zu", inputsSize);
  mtext += cnt;
  size  -= cnt;
  for (unsigned i = 0; i < inputsSizes.size(); ++i)
  {
    cnt = snprintf(mtext, size, ",%zu", inputsSizes[i]);
    mtext += cnt;
    size  -= cnt;
  }
  if (size == 0)
  {
    logging::critical("mtext buffer is too small for Inputs message");
    abort();
  }

  rc = _send(_inpMqId, msg, sizeof(msg) - size);
  if (rc)  return rc;

  char recvmsg[520];
  logging::info("[C++] Waiting for aknowledgment from Python");

  rc = _recv(_resMqId, recvmsg, sizeof(recvmsg), 5000);
  if (rc)  return rc;

  rc = _checkPy(_pyPid);
  if (rc)  return rc;

  // Provide Results shared memory info to python
  logging::info("[C++] Sending Results shared memory info to Python");

  mtext = &msg[0];
  size = sizeof(msg);
  msg[0] = 'r';
  cnt = 1;
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%s", ("/shmtebres_" + _keyBase).c_str());
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%zu", resultsSize);
  mtext += cnt;
  size  -= cnt;
  if (size == 0)
  {
    logging::critical("mtext buffer is too small for Results message");
    abort();
  }
  rc = _send(_inpMqId, msg, sizeof(msg) - size);
  if (rc)  return rc;

  rc = _checkPy(_pyPid);
  if (rc)  return rc;

  // Send connect message
  logging::info("[C++] Sending connect message to Python");
  msg[0] = 'c';
  msg[1] = ',';
  mtext = &msg[2];
  size  = sizeof(msg) - 2;
  auto cm  = _connectMsg.c_str();
  auto csz = _connectMsg.length();
  inpData = _inpData[0];
  while(csz > 0)
  {
    if (inputsSize > csz)  inputsSize = csz;
    memcpy(inpData, cm, inputsSize);
    cm     += inputsSize;
    csz    -= inputsSize;

    cnt = snprintf(mtext, size, "%zu", inputsSize);

    if (csz == 0)  msg[0] = 'd';  // Done

    rc = _send(_inpMqId, msg, cnt + 2); // Compensate for '<cmd>,' not in size
    if (rc)  return rc;

    rc = _checkPy(_pyPid);
    if (rc)  return rc;

    char recvmsg[520];
    char * recvmtext = &recvmsg[0];

    // Wait for the python side to finish initialization
    rc = _recv(_resMqId, recvmsg, sizeof(recvmsg), 5000);
    if (rc)  return rc;

    if (recvmtext[0] == 'd')  break;
    else if (recvmtext[0] != 'c')
      logging::error("Received error from Python: msg '%c'", recvmtext[0]);
  }

  return rc;
}

void Pds::Trg::TebPyTrig::event(const Pds::EbDgram* const* start,
                                const Pds::EbDgram**       end,
                                Pds::Eb::ResultDgram&      result)
{
  // Allow only L1Accept or Configure
   XtcData::TransitionId::Value transitionId = result.service();
   if (transitionId != XtcData::TransitionId::L1Accept &&
       transitionId != XtcData::TransitionId::Configure)
     return;

  *(Pds::Eb::ResultDgram*)_resData = result;

  uint64_t rem = (1ull<<_inpData.size())-1;
  const Pds::EbDgram* const* ctrb = start;
  do
  {
    auto dg   = *ctrb;
    auto size = sizeof(*dg) + dg->xtc.sizeofPayload();
    auto idx = dg->xtc.src.value();
    auto dest = _inpData[idx];
    rem ^= 1ull<<idx;
    memcpy(dest, dg, size);
  }
  while(++ctrb != end);

  // bit mask of contributions
  uint64_t mctrb = ((1ull<<_inpData.size())-1) ^ rem;

  int       rc;
  char msg[512];
  //  msg[0] = 'g';
  sprintf(msg,"g%016lx",mctrb);
  rc = _send(_inpMqId, msg, 17);

  if (rc == 0)
    rc = _checkPy(_pyPid);

  char recvmsg[520];
  if (rc == 0)
    rc = _recv(_resMqId, recvmsg, sizeof(recvmsg), 5000);

  if (rc == 0)
    if (recvmsg[0] != 'g')
      logging::error("Received error from Python: msg '%c'", recvmsg[0]);

  result = *(Pds::Eb::ResultDgram*)_resData;
}

void Pds::Trg::TebPyTrig::transition(Pds::Eb::ResultDgram& result)
{
  *(Pds::Eb::ResultDgram*)_resData = result;
    
}

// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::TebPyTrig;
}
