#include "SimDetector.hh"
#include "drp/XpmInfo.hh"               // For connectionInfo()
#include "psalg/utils/SysLog.hh"
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

  connectionShutdown();
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
  logging::debug("SimDetector::connect");

  m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

  std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
  if (it != m_para->kwargs.end())
    m_length = stoi(it->second);

  logging::debug("SimDetector::connect: end");
}

void SimDetector::issuePhase2(TransitionId::Value tid)
{
  logging::debug("SimDetector::issuePhase2");
  m_eventQueue.push(tid);
  logging::debug("SimDetector::issuePhase2 done");
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

void SimDetector::_trigger(CUdeviceptr dmaBuffer, uint32_t dmaSize) const
{
  // Write the dmaSize to the 2nd word of the DMA buffer
  chkError(cudaMemcpyAsync((void*)(dmaBuffer + sizeof(uint32_t)), &dmaSize, sizeof(dmaSize), cudaMemcpyHostToDevice, m_stream));
}

void SimDetector::_eventSimulator()
{
  logging::info("Event simulator is starting with process ID %lu\n", syscall(SYS_gettid));
  if (prctl(PR_SET_NAME, "drp_gpu/EvtSim", 0, 0, 0) == -1) {
    perror("prctl");
  }

  auto& memPool = *m_pool->getAs<MemPoolGpu>();
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

  bool running{false};
  auto suTime{fast_monotonic_clock::now()};
  auto swFpgaRegs{memPool.panels()[0].swFpgaRegs.ptr};
  unsigned dmaIdx{0};
  uint32_t dmaCntMsk{memPool.dmaCount() - 1};
  std::vector<uint8_t> buffer(sizeof(TimingHeader) + m_length * sizeof(uint32_t));
  auto dmaBuffers{memPool.panels()[0].dmaBuffers};
  while (!m_terminate.load(std::memory_order_acquire)) {
    // Wait for a new transition to appear from the "control level"
    TransitionId::Value tid;
    if (!m_eventQueue.pop(tid))  continue;
    //printf("*** _eventSimulator: Got tid %u (%s)\n", tid, TransitionId::name(tid));

    // Wait for DMA buffer to become ready for writing
    //printf("*** _eventSimulator: Waiting for DMA[%u] (%p) to become ready for writing\n", dmaIdx, swFpgaRegs + GPU_ASYNC_WR_ENABLE(dmaIdx));
    unsigned us = 1;
    while (!*(volatile uint8_t*)(swFpgaRegs + GPU_ASYNC_WR_ENABLE(dmaIdx))) {
      if (m_terminate.load(std::memory_order_acquire))  break;
      usleep(us);
      if (us < 1024)  us *= 2;
    }
    if (m_terminate.load(std::memory_order_acquire))  break;
    //printf("*** _eventSimulator: DMA[%u] (%p) is ready for writing\n", dmaIdx, swFpgaRegs + GPU_ASYNC_WR_ENABLE(dmaIdx));

    auto dmaBuffer = dmaBuffers[dmaIdx].dptr;
    uint32_t hndShk;
    chkError(cudaMemcpyAsync(&hndShk, (void*)(dmaBuffer + sizeof(uint32_t)), sizeof(hndShk), cudaMemcpyDeviceToHost, m_stream));
    //chkError(cudaStreamSynchronize(m_stream));
    //printf("*** _eventSimulator: DMA[%u]: wr enable %u, handshake %u\n", dmaIdx,
    //       *(volatile uint8_t*)(swFpgaRegs + GPU_ASYNC_WR_ENABLE(dmaIdx)), hndShk);
    auto dmaSize = _genTimingHeader(&buffer[0], tid);
    if (tid == TransitionId::L1Accept) {
      dmaSize += _genL1Payload(&buffer[dmaSize], buffer.size() - dmaSize);
    }
    chkError(cudaMemcpyAsync((void*)(dmaBuffer + sizeof(DmaDsc)), buffer.data(), dmaSize, cudaMemcpyHostToDevice, m_stream));
    _trigger(dmaBuffer, dmaSize);
    //printf("*** _eventSimulator: DMA[%u] triggered @ %p with %zu\n", dmaIdx, (void*)(dmaBuffer + sizeof(uint32_t)), dmaSize);
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
