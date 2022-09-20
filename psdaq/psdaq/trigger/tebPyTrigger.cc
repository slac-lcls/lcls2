#include "Trigger.hh"

#include "utilities.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/fast_monotonic_clock.hh"

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

using namespace rapidjson;
using json = nlohmann::json;
using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;


namespace Pds {
  namespace Trg {

    struct Message_t
    {
      long mtype;                       // message type
      char mtext[512];                  // message text
    };

    class TebPyTrig : public Trigger
    {
      static const unsigned KEY_BASE = 300000;
    public:
      TebPyTrig();
      virtual ~TebPyTrig();
      unsigned rogReserve(unsigned rog,
                          unsigned meb,
                          size_t   nBufs) const;
      int  configure(const json&              connectMsg,
                     const Document&          top,
                     const Pds::Eb::EbParams& prms) override;
      int  initialize(const std::vector<size_t>& inputsRegSize,
                      size_t                     resultsRegSize) override;
      void event(const Pds::EbDgram* const* start,
                 const Pds::EbDgram**       end,
                 Pds::Eb::ResultDgram&      result) override;
      void shutdown() override;
      void cleanup();
    private:
      void _drainPipe(int pipeFd);
      int  _startPython(pid_t& pyPid);
      int  _setupMsgQueue(key_t key, const char* name, int& id);
      int  _setupShMem(key_t       key,
                       size_t      size,
                       const char* name,
                       int&        id,
                       void*&      data);
      int _checkPy(pid_t, bool wait = false);
      int _send(int mqId, const Message_t&, size_t);
      int _recv(int mqId, Message_t&, unsigned msTmo);
    private:
      std::string _connectMsg;
      std::string _pythonScript;
      unsigned    _rogRsrvdBuf[Pds::Eb::NUM_READOUT_GROUPS];
    private:
      unsigned           _keyBase;
      unsigned           _partition;
      pid_t              _pyPid;
      int                _inpMqId;
      int                _resMqId;
      int                _inpShmId;
      int                _resShmId;
      std::vector<void*> _inpData;
      void*              _resData;
      int                _pipefd_stdout[2];
      int                _pipefd_stderr[2];
    };
  };
};


static Pds::Trg::TebPyTrig* _tebPyTrigger = nullptr;

static void cleanupOnSignal(int signum)
{
  logging::info("[C++] Cleaning up on receiving signal (%d)\n", signum);

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
      Message_t msg;
      msg.mtype    = 1;
      msg.mtext[0] = 's';
      rc = _send(_inpMqId, msg, 1);
      if (rc == 0)
        if (_pyPid)  _checkPy(_pyPid, true);
    }

    _drainPipe(_pipefd_stdout[0]);
    _drainPipe(_pipefd_stderr[0]);

    cleanup();

    close(_pipefd_stdout[0]);
    close(_pipefd_stderr[0]);

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
                                   const Document&          top,
                                   const Pds::Eb::EbParams& prms)
{
  int rc = 0;

  _connectMsg = connectMsg.dump();

# define _FETCH(key, item, type)                                        \
  if (top.HasMember(key))  item = top[key].type();                      \
  else { fprintf(stderr, "%s:\n  Key '%s' not found\n",                 \
                 __PRETTY_FUNCTION__, key);  rc = -1; }

  _FETCH("pythonScript", _pythonScript, GetString);

  for (unsigned rog = 0; rog < Pds::Eb::NUM_READOUT_GROUPS; ++rog)
  {
    char key[40];
    snprintf(key, sizeof(key), "rogRsrvdBuf[%u]", rog);
    _FETCH(key, _rogRsrvdBuf[rog], GetInt);
  }

# undef _FETCH

  auto scriptPath = prms.kwargs.find("script_path") != prms.kwargs.end()
                  ? const_cast<Pds::Eb::EbParams&>(prms).kwargs["script_path"]
                  : ".";                // Revisit: Good default?
  _pythonScript = scriptPath + "/" + _pythonScript;
  _partition    = prms.partition;

  _keyBase  = KEY_BASE + 10000 * prms.partition;
  _inpMqId  = 0;
  _resMqId  = 0;
  _inpShmId = 0;
  _inpData  .clear();
  _resShmId = 0;
  _resData  = nullptr;

  return rc;
}

void Pds::Trg::TebPyTrig::_drainPipe(int pipe)
{

  char read_buffer[1024];
  int nbytes;

  int flags = fcntl(pipe, F_GETFL);
  flags |= O_NONBLOCK;
  int ret_val = fcntl(pipe, F_SETFL, flags);
  if (ret_val < 0)
  {
    logging::error("fcntl failed: %m");
    return;
  }

  while (true)
  {

    nbytes = read(pipe, read_buffer, sizeof(read_buffer));
    if (nbytes > 0)
    {
      read_buffer[nbytes] = '\0';
      logging::info("%s", read_buffer);
    }
    else if (nbytes == 0)
    {
      return;
    }
    else if (nbytes < 0 && (errno == EWOULDBLOCK || errno == EAGAIN))
    {
      return;
    }
    else
    {
      logging::error("[C++] Error reading from pipe: %s: %m", read_buffer);
      logging::error("[C++] Exit error %d: %m", nbytes);
      return;
    }
  }
}

int Pds::Trg::TebPyTrig::_startPython(pid_t& pyPid)
{
  // Set up pipes for communication with sub processes
  if (pipe(_pipefd_stdout) == -1 || pipe(_pipefd_stderr) == -1)
  {
    logging::error("[C++]: Error creating stdout and stderr pipes");
    return -1;
  }

  // Fork
  pyPid = fork();

  if (pyPid == pid_t(0))
  {
    // Set up pipes in child process
    dup2(_pipefd_stdout[1], 1);
    dup2(_pipefd_stderr[1], 2);
    close(_pipefd_stdout[0]);
    close(_pipefd_stdout[1]);
    close(_pipefd_stderr[0]);
    close(_pipefd_stderr[1]);

    logging::info("Running 'python -u %s'", _pythonScript.c_str());

    // Executing external code - exec returns only on error
    int rc = execlp("python",
                    "python",
                    "-u",
                    _pythonScript.c_str(),
                    ("-p " + std::to_string(_partition)).c_str(),
                    nullptr);
    logging::error("Error on exec 'python -u %s': %m", _pythonScript);
    return rc;
  }
  return 0;
}

int Pds::Trg::TebPyTrig::_setupMsgQueue(key_t       key,
                                        const char* name,
                                        int&        mqId)
{
  mqId = msgget(key, IPC_CREAT | 0666);
  if (mqId == -1)
  {
    logging::error("[C++] Error in creating '%s' message queue with key %u: %m",
                   name, key);
    cleanup();
    return -1;
  }

  logging::info("[C++] '%s' message queues created", name);

  return 0;
}

int Pds::Trg::TebPyTrig::_setupShMem(key_t       key,
                                     size_t      size,
                                     const char* name,
                                     int&        shmId,
                                     void*&      data)
{
  shmId = shmget(key, size, IPC_CREAT | 0666); // IPC_EXCL
  if (shmId == -1)
  {
    logging::error("[C++] Error in creating '%s' shared memory for key %u: %m",
                   name, key);
    cleanup();
    return -1;
  }

  data = shmat(shmId, nullptr, 0);
  if (data == (void *)-1)
  {
    logging::error("[C++] Error attaching '%s' shared memory for key %u: %m",
                   name, key);
    cleanup();
    return -1;
  }

  logging::info("[C++] '%s' shared memory created for key %u", name, key);

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

int Pds::Trg::TebPyTrig::_send(int mqId, const Message_t& msg, size_t size)
{
  int rc = msgsnd(mqId, (void *)&msg, size, 0);
  if (rc == -1)
  {
    logging::error("[C++] Error sending message '%c': %m",
                   msg.mtext[0]);
    cleanup();
    return -1;
  }
  return 0;
}

int Pds::Trg::TebPyTrig::_recv(int mqId, Message_t& msg, unsigned msTmo)
{
  auto tp = Pds::fast_monotonic_clock::now();
  while (true)
  {
    auto rc = msgrcv(mqId, &msg, sizeof(msg.mtext), 0, IPC_NOWAIT);
    if (rc != -1)  break;

    if (errno != ENOMSG)
    {
      logging::error("[C++] Error receiving message: %m");
      cleanup();
      return -1;
    }

    _drainPipe(_pipefd_stdout[0]);
    _drainPipe(_pipefd_stderr[0]);

    auto now = Pds::fast_monotonic_clock::now();
    auto dt  = std::chrono::duration_cast<ms_t>(now - tp).count();
    if (dt > msTmo)
    {
      logging::error("[C++] Message receiving timed out");
      return -1;
    }
  }
  return 0;
}

void Pds::Trg::TebPyTrig::cleanup()
{
  if (_inpMqId)   { msgctl(_inpMqId, IPC_RMID, NULL);   _inpMqId  = 0; }
  if (_resMqId)   { msgctl(_resMqId, IPC_RMID, NULL);   _resMqId  = 0; }
  if (_inpData.size() &&                // Only [0] points to the real shmem
      _inpData[0]){ shmdt(_inpData[0]);                 _inpData[0] = nullptr; }
  if (_inpShmId)  { shmctl(_inpShmId, IPC_RMID, NULL);  _inpShmId = 0; }
  if (_resData)   { shmdt (_resData);                   _resData  = nullptr; }
  if (_resShmId)  { shmctl(_resShmId, IPC_RMID, NULL);  _resShmId = 0; }
  _drainPipe(_pipefd_stdout[0]);
  _drainPipe(_pipefd_stderr[0]);
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

  // Set up pipes in parent process
  close(_pipefd_stdout[1]);
  close(_pipefd_stderr[1]);

  // Creating message queues
  logging::info("[C++] Creating message queues");

  rc = _setupMsgQueue(_keyBase+0, "Inputs", _inpMqId);
  if (rc)  return rc;

  rc = _setupMsgQueue(_keyBase+1, "Results", _resMqId);
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
  rc = _setupShMem(_keyBase+2, inputsSize, "Inputs", _inpShmId, inpData);
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

  rc = _setupShMem(_keyBase+3, resultsSize, "Results", _resShmId, _resData);
  if (rc)  return rc;

  // Provide Inputs shared memory info to python
  logging::info("[C++] Sending Inputs shared memory info to Python");

  Message_t msg;
  char*     mtext = msg.mtext;
  size_t    size = sizeof(msg.mtext);
  msg.mtype    = 1;
  msg.mtext[0] = 'i';
  int cnt = 1;
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%u", _keyBase+2);
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
  rc = _send(_inpMqId, msg, sizeof(msg.mtext) - size);
  if (rc)  return rc;

  rc = _checkPy(_pyPid);
  if (rc)  return rc;

  // Provide Results shared memory info to python
  logging::info("[C++] Sending Results shared memory info to Python");

  mtext = msg.mtext;
  size  = sizeof(msg.mtext);
  msg.mtype    = 1;
  msg.mtext[0] = 'r';
  cnt = 1;
  mtext += cnt;
  size  -= cnt;
  cnt = snprintf(mtext, size, ",%u", _keyBase+3);
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
  rc = _send(_inpMqId, msg, sizeof(msg.mtext) - size);
  if (rc)  return rc;

  rc = _checkPy(_pyPid);
  if (rc)  return rc;

  // Send connect message
  logging::info("[C++] Sending connect message to Python");
  msg.mtype    = 1;
  msg.mtext[0] = 'c';
  msg.mtext[1] = ',';
  mtext = &msg.mtext[2];
  size  = sizeof(msg.mtext) - 2;
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

    if (csz == 0)  msg.mtext[0] = 'd';  // Done

    rc = _send(_inpMqId, msg, cnt + 2); // Compensate for '<cmd>,' not in size
    if (rc)  return rc;

    rc = _checkPy(_pyPid);
    if (rc)  return rc;

    // Wait for the python side to finish initialization
    rc = _recv(_resMqId, msg, 5000);
    if (rc)  return rc;

    _drainPipe(_pipefd_stdout[0]);
    _drainPipe(_pipefd_stderr[0]);

    if (msg.mtext[0] == 'd')  break;
    else if (msg.mtext[0] != 'c')
      logging::error("Received error from Python: msg '%c'", msg.mtext[0]);
  }

  return rc;
}

void Pds::Trg::TebPyTrig::event(const Pds::EbDgram* const* start,
                                const Pds::EbDgram**       end,
                                Pds::Eb::ResultDgram&      result)
{
  _drainPipe(_pipefd_stdout[0]);
  _drainPipe(_pipefd_stderr[0]);

  *(Pds::Eb::ResultDgram*)_resData = result;

  unsigned idx = 0;
  const Pds::EbDgram* const* ctrb = start;
  do
  {
    auto dg   = *ctrb;
    auto size = sizeof(*dg) + dg->xtc.sizeofPayload();
    auto dest = _inpData[idx++];
    memcpy(dest, dg, size);
  }
  while(++ctrb != end);

  if (idx < _inpData.size())            // zero terminate
    *(EbDgram*)(_inpData[idx]) = EbDgram(PulseId{0}, XtcData::Dgram());

  int       rc;
  Message_t msg;
  msg.mtype    = 1;
  msg.mtext[0] = 'g';
  rc = _send(_inpMqId, msg, 1);

  if (rc == 0)
    rc = _checkPy(_pyPid);

  if (rc == 0)
    rc = _recv(_resMqId, msg, 5000);

  if (rc == 0)
    if (msg.mtext[0] != 'g')
      logging::error("Received error from Python: msg '%c'", msg.mtext[0]);

  result = *(Pds::Eb::ResultDgram*)_resData;
}


// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::TebPyTrig;
}
