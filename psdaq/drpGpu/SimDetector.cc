#include "SimDetector.hh"
#include "drp/XpmInfo.hh"               // For connectionInfo()
#include "psalg/utils/SysLog.hh"

#include <time.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/prctl.h>

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json = nlohmann::json;
using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;

static const unsigned EvtCtrMask = 0xffffff;

[[maybe_unused]]
static int _nsSleep(unsigned ns)
{
  struct timespec ts{0, ns};
  return nanosleep(&ts, nullptr);
}


SimDetector::SimDetector(Parameters* para, MemPoolGpu* pool, unsigned len) :
  Detector      (para, pool),
  m_terminate   (false),
  m_stream      (0),
  m_readoutGroup(para->partition),
  m_evtCounter  (0),
  m_length      (len),
  m_eventQueue  (pool->dmaCount()),
  m_l1Delay     (1000000),              // L1Accept delay in us
  m_suPeriod    {1000}                  // SlowUpdate period in ms
{
  if (para->kwargs.find("sim_l1_delay") != para->kwargs.end()) {
    m_l1Delay = std::stoul(para->kwargs["sim_l1_delay"]);
  }
  if (para->kwargs.find("sim_su_rate") != para->kwargs.end()) {
    auto rate = std::stof(para->kwargs["sim_su_rate"]);
    m_suPeriod = ms_t{rate ? unsigned(1000./rate) : 0};
  }

  printf("*** SimDetector::ctor: this %p, len %u, l1Delay %u\n", this, m_length, m_l1Delay);
}

SimDetector::~SimDetector()
{
  printf("*** SimDetector::dtor: 1\n");

  if (m_stream) {
    chkError(cudaStreamDestroy(m_stream));
  }

  printf("*** SimDetector::dtor: 2\n");
  shutdown();

  printf("*** SimDetector::dtor: 3\n");
}

void SimDetector::shutdown()
{
  logging::debug("SimDetector::shutdown");
}

json SimDetector::connectionInfo(const json& msg)
{
  logging::debug("SimDetector::connectionInfo");

  // Reset the event queue
  m_eventQueue.startup();

  m_terminate.store(false, std::memory_order_release);

  // Start the event simulator
  m_eventThread = std::thread(&SimDetector::_eventSimulator, std::ref(*this));

  logging::debug("SimDetector::connectionInfo: end");

  json result = xpmInfo(-1u);
  return result;
}

void SimDetector::connectionShutdown()
{
  logging::debug("SimDetector::connectionShutdown");

  // Unblock the record queue
  m_eventQueue.shutdown();

  m_terminate.store(true, std::memory_order_release);

  if (m_eventThread.joinable()) {
    m_eventThread.join();
    logging::info("Event simulator thread finished");
  }
}

// setup up device to receive data over pgp
void SimDetector::connect(const json& connect_json, const std::string& collectionId)
{
  logging::debug("SimDetector::connect: this %p", this);

  m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

  std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
  if (it != m_para->kwargs.end())
    m_length = stoi(it->second) * sizeof(uint32_t); // Convert to bytes

  logging::debug("SimDetector::connect: end this %p", this);
}

void SimDetector::issuePhase2(TransitionId::Value tid)
{
  logging::debug("SimDetector::issuePhase2: this %p, tid %s", this, TransitionId::name(tid));
  m_eventQueue.push(tid);
}

size_t SimDetector::_genTimingHeader(uint8_t* buffer, TransitionId::Value tid)
{
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec -= POSIX_TIME_AT_EPICS_EPOCH;
  uint64_t pid{uint64_t(ts.tv_sec) * 14000000/13 + (uint64_t(ts.tv_nsec) * 14)/13000};
  auto th = (TimingHeader*)buffer;
  *(uint64_t*)th = (uint64_t(tid)<<56) | (pid & ((1ul << 56) - 1)); // PulseId
  th->time       = TimeStamp(ts);
  th->env        = 1 << m_readoutGroup;
  th->evtCounter = m_evtCounter;
  th->_opaque[0] = 0;
  th->_opaque[1] = 0;

  m_evtCounter = (m_evtCounter + 1) & EvtCtrMask;
  return sizeof(*th);
}

void SimDetector::_trigger(uint8_t* dmaBuffer, uint32_t dmaSize) const
{
  DmaDsc dmaDsc{0, dmaSize};

  // Write the dmaSize to the 2nd word of the DMA buffer
  chkError(cudaMemcpyAsync((void*)dmaBuffer, &dmaDsc, sizeof(dmaDsc), cudaMemcpyDefault, m_stream));
}

void SimDetector::_eventSimulator()
{
  logging::info("Event simulator is starting with process ID %lu\n", syscall(SYS_gettid));
  if (prctl(PR_SET_NAME, "drp_gpu/EvtSim", 0, 0, 0) == -1) {
    perror("prctl");
  }

  auto& memPool = *m_pool->getAs<MemPoolGpu>();
  chkError(cudaSetDevice(memPool.context().deviceNo()));
  chkError(cuCtxSetCurrent(memPool.context().context()));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo};
  logging::debug("Event simulator stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create a GPU stream in the event simulator thread context
  // The lowest priority is to inject more events into the system
  chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));

  bool verify{m_para->kwargs.find("sim_l1_verify") != m_para->kwargs.end()};
  bool running{false};
  auto suTime{fast_monotonic_clock::now()};
  auto panel{memPool.panel()};
  auto fpgaRegs{panel->fpgaRegs.h};
  auto& coreRegs{panel->coreRegs};
  unsigned dmaIdx{0};
  uint32_t dmaCntMsk{memPool.dmaCount() - 1};
  std::vector<uint8_t> thBuffer(sizeof(TimingHeader)); // No default constructor for TimingHeader
  auto thOffset{coreRegs.dmaDataBytes()};
  auto dmaBuffers{panel->dmaBuffers};
  while (!m_terminate.load(std::memory_order_acquire)) {
    // Wait for a new transition to appear from the "control level"
    TransitionId::Value tid;
    if (!m_eventQueue.pop(tid))  continue;

    // Reset the DMA buffer index to be consistent with Reader::_handleDMA()
    if (tid == TransitionId::Configure) {
      dmaIdx       = 0;
      m_evtCounter = 0;
    } else if (tid == TransitionId::BeginRun) {
      m_evtCounter = 1;                 // Compensate for the ClearReadout sent before BeginRun
    }

    // Wait for DMA buffer to become ready for writing
    //bool wait{false};
    //unsigned ns{8};
    volatile uint8_t* const wrEnReg{(uint8_t*)fpgaRegs + coreRegs.freeListOffset(dmaIdx)};
    //uint32_t hndShk;
    auto dmaBuffer = dmaBuffers[dmaIdx];
    //printf("*** SimDetector::evtSim: dmaIdx %u, dmaBuffer %p\n", dmaIdx, dmaBuffer);
    //chkError(cudaMemcpyAsync(&hndShk, (void*)(dmaBuffer + sizeof(uint32_t)), sizeof(hndShk), cudaMemcpyDefault, m_stream));
    //chkError(cudaStreamSynchronize(m_stream));
    //printf("*** SimDetector::evtSim: Wait for wrEnReg[%u] %p\n", dmaIdx, wrEnReg);
    while (*wrEnReg == 0) { // || (hndShk != 0)) {
      if (m_terminate.load(std::memory_order_acquire))
        break;
      //_nsSleep(ns);
      //if (ns < 256)  ns *= 2;
      //if (!wait) {
      //  wait = true;
      //  printf("*** SimDetector::evtSim: wait T, dmaIdx %u, evtCtr %u, hndShk %u\n", dmaIdx, m_evtCounter, hndShk);
      //}
      //chkError(cudaMemcpyAsync(&hndShk, (void*)(dmaBuffer + sizeof(uint32_t)), sizeof(hndShk), cudaMemcpyDefault, m_stream));
      //chkError(cudaStreamSynchronize(m_stream));
    }
    if (m_terminate.load(std::memory_order_acquire))
      break;
    //if (wait) {
    //  wait = false;
    //  printf("*** SimDetector::evtSim: wait F, dmaIdx %u, evtCtr %u, hndShk %u\n", dmaIdx, m_evtCounter, hndShk);
    //}

    asm volatile("mfence" ::: "memory");
    *wrEnReg = 0;                       // Disable "DMAs" to this buffer

    auto evtIdx = m_evtCounter;         // Fetch this before it's incremented in _genTimingHeader
    auto thSize = _genTimingHeader(thBuffer.data(), tid);
    //uint32_t*p = (uint32_t*)thBuffer.data();
    //printf("*** SimDetector::evtSim: thBuf %08x %08x %08x %08x %08x %08x %08x %08x\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    size_t dmaSize{thSize};
    if (tid == TransitionId::L1Accept) {
      auto hdrSize{thOffset + thSize};
      uint8_t* data_d;
      auto dataSize = _genL1Payload(&data_d, evtIdx, 0); // Don't do partial events for now: m_length);
      if (verify) {  // Avoid overhead by not injecting data if it's not going to be verified
        chkError(cudaMemcpyAsync((void*)(dmaBuffer + hdrSize), data_d, dataSize, cudaMemcpyDefault, m_stream));
      }
      dmaSize += dataSize;
    }
    //printf("*** SimDetector::evtSim: dmaBuf %p + thOs %u = %p\n", dmaBuffer, thOffset, dmaBuffer + thOffset);
    chkError(cudaMemcpyAsync((void*)(dmaBuffer + thOffset), thBuffer.data(), thSize, cudaMemcpyDefault, m_stream));
    _trigger(dmaBuffer, dmaSize);
    dmaIdx = (dmaIdx + 1) & dmaCntMsk;

    if (running) {
      auto t{fast_monotonic_clock::now()};
      if ((m_suPeriod > ms_t{0}) && (t - suTime > m_suPeriod)) {
        m_eventQueue.push(TransitionId::SlowUpdate);
        suTime = t;
      }
    }

    // Prepare next event
    switch (tid) {
      case TransitionId::Enable:
        if (m_suPeriod > ms_t{0})  m_eventQueue.push(TransitionId::SlowUpdate);
        m_eventQueue.push(TransitionId::L1Accept);
        running = true;
        break;
      case TransitionId::Disable:
        running = false;
        break;
      case TransitionId::L1Accept:
        if (running) {
          if (m_l1Delay)  usleep(m_l1Delay);
          m_eventQueue.push(TransitionId::L1Accept);
        }
        break;
      default:
        break;
    }
  }

  logging::info("Event simulator thread is exiting");
}
