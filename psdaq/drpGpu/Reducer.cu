#include "Reducer.hh"
#include "ReducerAlgo.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace Drp;
using namespace Drp::Gpu;


Reducer::Reducer(const Parameters& para, MemPoolGpu& pool) :
  m_pool   (pool),
  m_reducer(nullptr),
  m_para   (para)
{
  /** Create the Reducer stream **/
  chkFatal(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
  logging::debug("Done with creating stream");

  m_reducer = _setupReducer();
  if (!m_reducer) {
    logging::critical("Error setting up Reducer");
    abort();
  }
}

Reducer::~Reducer()
{
  chkError(cudaStreamDestroy(m_stream));

  if (m_reducer)  delete m_reducer;
}

ReducerAlgo* Reducer::_setupReducer()
{
  // @todo: In the future, find out which Reducer to load from the Detector's configDb entry
  //        For now, load it according to a command line kwarg parameter
  std::string reducer;
  if (m_para.kwargs.find("reducer") == m_para.kwargs.end()) {
    logging::error("Missing required kwarg 'reducer'");
    return nullptr;
  }
  reducer = m_para.kwargs["reducer"];

  if (m_reducer)  delete m_reducer;     // If the object exists, delete it
  m_dl.close();                         // If a lib is open, close it first

  const std::string soName("lib"+reducer+".so");
  if (m_dl.open(soName, RTLD_LAZY)) {
    logging::error("Error opening library '%s'", soName.c_str());
    return nullptr;
  }
  const std::string symName("createReducer");
  auto createFn = m_dl.loadSymbol(symName.c_str());
  if (!createFn) {
    logging::error("Symbol '%s' not found in %s",
                   symName.c_str(), soName.c_str());
    return nullptr;
  }
  auto instance = reinterpret_cast<reducerAlgoFactoryFn_t*>(createFn)(m_para, m_pool);
  if (!instance)
  {
    logging::error("Error calling %s from %s", symName.c_str(), soName.c_str());
    return nullptr;
  }
  return instance;
}

int Reducer::_setupGraph()
{
  return 0;
}

void Reducer::start(Detector* det, GpuMetrics& metrics)
{
  m_terminate_h.store(false, std::memory_order_release);

  // Launch the forwarding thread
  m_thread = std::thread(&Reducer::_reduce, std::ref(*this),
                         std::ref(*det), std::ref(metrics));
}

void Reducer::stop()
{
  logging::warning("Gpu::Reducer::stop called");

  chkError(cuCtxSetCurrent(m_pool.context().context()));

  // Stop and clean up the streams
  m_terminate_h.store(true, std::memory_order_release);

  logging::info("Shutting down Reducer stream");
  __sync_fetch_and_add(m_terminate_d, 1);
}

void Reducer::_reduce()
{
  const uint32_t bufferMask = m_ringIndex_h->size() - 1;
  unsigned tail = 0;
  unsigned head = 0;

  uint64_t lastPid = 0;
  while (!m_terminate_h.load(std::memory_order_acquire)) {
    tail = head;
    head = m_ringIndex_h->consume();
    unsigned index = tail;
    while (index != head) {
      unsigned nDmaRet = head - index;
      const volatile auto dsc = (DmaDsc*)(m_hostWriteBufs[index]);
      const volatile auto th  = (TimingHeader*)&dsc[1];

      metrics.m_nDmaRet.store(0);
      // Wait for the GPU to have processed an event
      //chkFatal(cuCtxSynchronize());
      //chkError(cuStreamSynchronize(m_dmaStreams[index]));
      uint64_t pid;
      while (!m_terminate_h.load(std::memory_order_acquire)) {
        pid = th->pulseId();
        if (pid > lastPid)  break;
        if (!lastPid && !pid)  break;   // Expect lastPid to be 0 only on startup
      }
      if (m_terminate_h.load(std::memory_order_acquire))  break;
      if (!pid)  continue;              // Search for a DMA buffer with data in it
      lastPid = pid;

      metrics.m_nDmaRet.store(nDmaRet);
      m_pool.allocateDma();

      logging::debug("*** dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x\n",
                     index, dsc->error, dsc->size, dsc->_rsvd[0], dsc->_rsvd[1], dsc->_rsvd[2],
                     dsc->_rsvd[3], dsc->_rsvd[4], dsc->_rsvd[5]);
      logging::debug("**G dma %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x\n",
                     index, th->control(), th->pulseId(), th->time.value(), th->env, th->evtCounter,
                     th->_opaque[0], th->_opaque[1]);

      uint32_t size = dsc->size;
      ///uint32_t lane = 0;                  // The lane is always 0 for GPU-enabled PGP devices
      metrics.m_dmaSize   = size;
      metrics.m_dmaBytes += size;
      // @todo: Is this the case here also?
      // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
      // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
      // which thus won't have the expected header.  Take the exact match as an overflow indicator.
      if (size == m_pool.dmaSize()) {
        logging::critical("%d DMA overflowed buffer: %d vs %d", index, size, m_pool.dmaSize());
        //abort();
      }

      const Pds::TimingHeader* timingHeader = th; //det.getTimingHeader(index);
      if (!timingHeader)  printf("*** No timingHeader at ctr %d\n", th->evtCounter);
      ///uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
      ///unsigned pgpIndex = evtCounter & bufferMask;
      ///PGPEvent*  event  = &m_pool.pgpEvents[pgpIndex];
      ///if (!event)  printf("*** No pgpEvent for ctr %d\n", pgpIndex);
      ///DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
      ///if (!buffer)  printf("*** No pgpEvent.buffer for lane %d, ctr %d\n", lane, pgpIndex);
      ///buffer->size = size;                       //   "
      ///buffer->index = index;                     //   "
      ///event->mask |= (1 << lane);

      if (dsc->error) {
        logging::error("DMA error 0x%x",dsc->error);
        ///handleBrokenEvent(*event);
        freeDma(index);                 // Leaves event mask = 0
        metrics.m_nDmaErrors += 1;
        continue;
      }

      if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
        metrics.m_nTmgHdrError += 1;
      }
      XtcData::TransitionId::Value transitionId = timingHeader->service();

      auto rogs = timingHeader->readoutGroups();
      if ((rogs & (1 << m_para.partition)) == 0) {
        logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId(), m_para.partition, timingHeader->env);
        ///handleBrokenEvent(*event);
        freeDma(index);                 // Leaves event mask = 0
        metrics.m_nNoComRoG += 1;
        continue;
      }

      // @todo: Shouldn't this check for missing RoGs on all transitions?
      if (transitionId == XtcData::TransitionId::SlowUpdate) {
        uint16_t missingRogs = m_para.rogMask & ~rogs;
        if (missingRogs) {
          logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId(), missingRogs, timingHeader->env);
          ///handleBrokenEvent(*event);
          freeDma(index);             // Leaves event mask = 0
          metrics.m_nMissingRoGs += 1;
          continue;
        }
      }

      const uint32_t* data = reinterpret_cast<const uint32_t*>(th);
      logging::debug("GpuWorker  size %u  hdr %016lx.%016lx.%08x  err 0x%x",
                     size,
                     reinterpret_cast<const uint64_t*>(data)[0], // PulseId
                     reinterpret_cast<const uint64_t*>(data)[1], // Timestamp
                     reinterpret_cast<const uint32_t*>(data)[4], // env
                     dsc->error);

      if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
          logging::info("GpuWorker  saw %s @ %u.%09u (%014lx)",
                        XtcData::TransitionId::name(transitionId),
                        timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                        timingHeader->pulseId());
        }
        else {
          logging::debug("GpuWorker  saw %s @ %u.%09u (%014lx)",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId());
        }
        if (transitionId == XtcData::TransitionId::BeginRun) {
          resetEventCounter();
        }
      }

      metrics.m_nevents += 1;
      m_dmaQueue.push(index);
      index = (index+1)&bufferMask;
    }
  }

  m_dmaQueue.shutdown();

  logging::debug("Gpu::Reducer exiting");
}
