#include "Reducer.hh"

#include "Detector.hh"
#include "PassthruShim.hh"              // Linked in, not loaded: see _setupAlgo()

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/eb/src/ResultDgram.hh"
#include "ReducerAlgo.hh"

using namespace XtcData;
using namespace Drp;
using namespace Drp::Gpu;
using namespace Pds;
using namespace Pds::Eb;

using logging = psalg::SysLog;
using json    = nlohmann::json;

struct red_domain{ static constexpr char const* name{"Reducer"}; };
using red_scoped_range = nvtx3::scoped_range_in<red_domain>;

// Prescale buffers for prescaled events.  The pebble pool bounds how many can be in
// flight -- the Reader cannot outrun the recorder by more than nbuffers() events,
// because pool.allocate() blocks -- which at a 1 Hz keepRaw rate and 33 kHz is a small
// fraction of one event.  32 is therefore enormous headroom, and cheap: each costs the
// same as one reduce buffer plus the raw block.
static constexpr unsigned NPrescaleBuffers{32};

static inline unsigned nxtPwrOf2(unsigned n)
{
  return n > 1 ? 1 << (32 - __builtin_clz(n - 1)) : 0;
}


Reducer::Reducer(const Parameters&                  para,
                 MemPoolGpu&                        pool,
                 Detector&                          det,
                 cudaExecutionContext_t             green_ctx,
                 const std::atomic<bool>&           terminate,
                 const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool       (pool),
  m_algo       (nullptr),
  m_terminate  (terminate),
  m_terminate_d(terminate_d),
  m_timePrev   {0, 0},
  m_para       (para)
{
  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioHi+1};
  logging::debug("Reducer stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create the Reducer streams
  m_streams.resize(m_para.nworkers);
  m_indices_d.resize(m_para.nworkers);
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    //chkFatal(cudaStreamCreateWithPriority(&m_streams[i], cudaStreamNonBlocking, prio));
    chkFatal(cudaExecutionCtxStreamCreate(&m_streams[i], green_ctx, cudaStreamNonBlocking, prio));

    // Index of the buffer being reduced; device memory since every reduce block reads it
    chkError(cudaMalloc(&m_indices_d[i],    sizeof(*m_indices_d[i])));
    chkError(cudaMemset( m_indices_d[i], 0, sizeof(*m_indices_d[i])));
  }
  logging::debug("Done with creating %u Reducer streams", m_streams.size());

  // Set up the reducer algorithm
  if (!_setupAlgo(det)) {
    logging::critical("Error setting up Reducer Algorithm");
    abort();
  }

  // The header consists of the Dgram with the parent Xtc, the ShapesData Xtc, the
  // Shapes Xtc with its payload and Data Xtc, the payload of which is on the GPU.
  auto shapesDataSize = 3 * sizeof(Xtc) + MaxRank * sizeof(uint32_t);
  auto headerSize     = sizeof(Dgram) + shapesDataSize;
  // Exactly what the algorithm asks for; writing more than that is its bug, which
  // the recorder checks for
  auto payloadSize = m_algo ? m_algo->payloadSize() : 0;

  // In pass-through the raw block *is* the recorded data and the reduced payload is
  // unused, which is why payloadSize() may legitimately be 0 there.  Only that mode
  // puts raw in the reduce buffers: a prescaled BEAM event assembles its datagram in
  // a prescale buffer instead, so the reduce buffers stay [hdr][reduced] and the raw
  // block is not multiplied by nbuffers().
  auto rawSize = det.rawSize();
  auto redRaw  = det.passthru() ? rawSize : 0;
  if (redRaw)
    logging::warning("Reserving %zu B per buffer for raw data ahead of the reduced payload",
                     redRaw);

  // Prepare buffers to receive the reduced data, prepended with reserved space for
  // the datagram header and, in pass-through, for raw data.
  // The application sees only the pointer to the data buffer.
  m_pool.createReduceBuffers(payloadSize, headerSize, redRaw);

  // Transitions get their own buffers so that the largest transition's size is not
  // multiplied by nbuffers().  The count comes from the CPU pool: all transitions but
  // SlowUpdate are synchronous, so only SlowUpdates -- 1 Hz, unacknowledged -- can
  // accumulate, and that count is many minutes' worth of them.
  m_pool.createTransitionBuffers(m_para.maxTrSize, m_pool.pebble.nTrBuffers());

  // Prescaled events need somewhere to assemble [hdr][raw][reduced] contiguously.
  // Only in BEAM: pass-through records raw for every event, through the reduce
  // buffers, so it needs none of these.  The count is generous because they are
  // nearly free -- 32 costs ~0.03 GiB where reserving raw in every reduce buffer
  // costs 0.74 GiB at nbuffers = 2048 -- and because running dry aborts the DAQ.
  // A prescaled event describes two arrays, so its datagram carries a second block of
  // descriptors; it goes between the payloads, not in the header reserve, because
  // CreateData writes each array's descriptors immediately before its bytes.
  if (rawSize && !det.passthru())
    m_pool.createPrescaleBuffers(rawSize, NPrescaleBuffers, headerSize, shapesDataSize);

  // Set up the worker queues to fit all buffers
  if (m_para.nworkers) {
    auto nEntries{nxtPwrOf2((m_pool.nbuffers() + m_para.nworkers-1) / m_para.nworkers)};
    m_inputQueues.resize(m_para.nworkers);
    m_outputQueues.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      auto& iq = m_inputQueues[i];
      iq.h = new RingQueueHtoD<unsigned>(nEntries);
      chkError(cudaMalloc(&iq.d,       sizeof(*iq.d)));
      chkError(cudaMemcpy( iq.d, iq.h, sizeof(*iq.d), cudaMemcpyHostToDevice));
      auto& oq = m_outputQueues[i];
      oq.h = new RingQueueDtoH<ReducerTuple>(nEntries);
      chkError(cudaMalloc(&oq.d,       sizeof(*oq.d)));
      chkError(cudaMemcpy( oq.d, oq.h, sizeof(*oq.d), cudaMemcpyHostToDevice));
    }

    // Where the algorithm reports a per-event problem; _reducerRcv zeroes it
    m_retCode_d.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      chkError(cudaMalloc(&m_retCode_d[i],    sizeof(*m_retCode_d[i])));
      chkError(cudaMemset( m_retCode_d[i], 0, sizeof(*m_retCode_d[i])));
    }

    m_time_d.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      chkError(cudaMalloc(&m_time_d[i],    sizeof(*m_time_d[i])));
      chkError(cudaMemset( m_time_d[i], 0, sizeof(*m_time_d[i])));
    }

    // Optional loop timing histograms, a debugging aid
    m_hist_d.resize(m_para.nworkers, nullptr);
    if (para.kwargs.find("readerHist") != para.kwargs.end() && para.kwargs.at("readerHist") != "0") {
      for (unsigned i = 0; i < m_para.nworkers; ++i) {
        chkError(cudaMalloc(&m_hist_d[i],    sizeof(*m_hist_d[i])));
        chkError(cudaMemset( m_hist_d[i], 0, sizeof(*m_hist_d[i])));
      }
    }

    // Set up a state variable
    m_state_d.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      chkError(cudaMalloc(&m_state_d[i],    sizeof(*m_state_d[i])));
      chkError(cudaMemset( m_state_d[i], 0, sizeof(*m_state_d[i])));
    }

    // Prepare metrics for tracking kernel state and execution progress
    m_metrics.state.resize(m_para.nworkers);
    m_metrics.inpWtCtr.resize(m_para.nworkers);
    m_metrics.outWtCtr.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      if (!m_metrics.state[i]) {
        chkError(hostAlloc(&m_metrics.state[i], sizeof(*m_metrics.state[i])));
      }
      *m_metrics.state[i] = 0;

      chkError(hostAlloc(&m_metrics.inpWtCtr[i], sizeof(*m_metrics.inpWtCtr[i])));
      *m_metrics.inpWtCtr[i] = 0;
      chkError(hostAlloc(&m_metrics.outWtCtr[i], sizeof(*m_metrics.outWtCtr[i])));
      *m_metrics.outWtCtr[i] = 0;
    }
  }
}

Reducer::~Reducer()
{
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_metrics.inpWtCtr[i]) {
      chkError(hostFree(m_metrics.inpWtCtr[i]));
      m_metrics.inpWtCtr[i] = nullptr;
    }
    if (m_metrics.outWtCtr[i]) {
      chkError(hostFree(m_metrics.outWtCtr[i]));
      m_metrics.outWtCtr[i] = nullptr;
    }

    if (m_metrics.state[i]) {
      hostFree(m_metrics.state[i]);
      m_metrics.state[i] = nullptr;
    }
  }
  m_metrics.inpWtCtr.clear();
  m_metrics.outWtCtr.clear();
  m_metrics.state.clear();

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_state_d[i])  chkError(cudaFree(m_state_d[i]));
    m_state_d[i] = nullptr;
  }
  m_state_d.clear();

  for (auto& hist : m_hist_d) {
    if (hist)  chkError(cudaFree(hist));
    hist = nullptr;
  }
  m_hist_d.clear();

  for (auto& time : m_time_d) {
    if (time)  chkError(cudaFree(time));
    time = nullptr;
  }
  m_time_d.clear();

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    if (m_retCode_d[i])  chkError(cudaFree(m_retCode_d[i]));
    m_retCode_d[i] = nullptr;
  }
  m_retCode_d.clear();

  for (unsigned i = 0; i < m_inputQueues.size(); ++i) {
    if (m_inputQueues[i].d)   chkError(cudaFree(m_inputQueues[i].d));
    if (m_inputQueues[i].h)   delete m_inputQueues[i].h;
    if (m_outputQueues[i].d)  chkError(cudaFree(m_outputQueues[i].d));
    if (m_outputQueues[i].h)  delete m_outputQueues[i].h;
  }
  m_inputQueues.clear();
  m_outputQueues.clear();

  for (auto& graphExec : m_graphExecs) {
    chkError(cudaGraphExecDestroy(graphExec));
  }
  m_graphExecs.clear();

  if (m_algo)  delete m_algo;
  m_dl.close();

  m_pool.destroyPrescaleBuffers();
  m_pool.destroyTransitionBuffers();
  m_pool.destroyReduceBuffers();

  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    chkError(cudaFree(m_indices_d[i]));

    chkError(cudaStreamDestroy(m_streams[i]));
  }
  m_indices_d.clear();
  m_streams.clear();
}

int Reducer::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                          std::map<std::string, std::string>&   labels)
{
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    auto wkr = std::to_string(i);
    exporter->add("DRP_redState"+wkr, labels, MetricType::Gauge, [&, i](){ return m_metrics.state[i] ? *m_metrics.state[i] : 0; });

    *m_metrics.inpWtCtr[i] = 0;
    *m_metrics.outWtCtr[i] = 0;
    exporter->add("DRP_inpWtCtr"+wkr, labels, MetricType::Counter, [&, i](){ return m_metrics.inpWtCtr[i] ? *m_metrics.inpWtCtr[i] : 0; });
    exporter->add("DRP_outWtCtr"+wkr, labels, MetricType::Counter, [&, i](){ return m_metrics.outWtCtr[i] ? *m_metrics.outWtCtr[i] : 0; });
  }

  for (unsigned i = 0; i < m_inputQueues.size(); ++i) {
    auto wkr = std::to_string(i);
    exporter->add("DRP_inputQueue"+wkr,  labels, MetricType::Gauge, [&, i](){ return m_inputQueues[i].h->occupancy(); });
    exporter->add("DRP_outputQueue"+wkr, labels, MetricType::Gauge, [&, i](){ return m_outputQueues[i].h->occupancy(); });
  }

  m_timePrev[0] = m_timePrev[1] = 0;
  exporter->addFloat("DRP_reduceTime", labels, [&](double& us){
    chkError(cuCtxSetCurrent(m_pool.context().context())); // The exporter's threads have none
    return stageTimeMean(m_time_d.data(), m_time_d.size(), m_timePrev, us);
  });

  return 0;
}

bool Reducer::_setupAlgo(Detector& det)
{
  if (m_algo)  delete m_algo;           // If the object exists, delete it
  m_dl.close();                         // If a lib is open, close it first

  // Pass-through needs no reduction algorithm, just the shim that reports the raw
  // block's size and completes each event.  It is linked in rather than loaded so
  // that the operator can switch between BEAM and CALIB from run to run without the
  // .cnf.py changing: the kwarg below names the reducer for BEAM either way.
  if (det.passthru()) {
    m_algo = new PassthruShim(m_para, m_pool, det);
    return true;
  }

  // @todo: In the future, find out which Reducer to load from the Detector's configDb entry
  //        For now, load it according to a command line kwarg parameter
  std::string reducer;
  if (m_para.kwargs.find("reducer") == m_para.kwargs.end()) {
    logging::error("Missing required kwarg 'reducer'");
    return false;
  }
  reducer = m_para.kwargs.at("reducer");

  const std::string soName("lib"+reducer+".so");
  logging::debug("Loading library '%s'", soName.c_str());
  if (m_dl.open(soName, RTLD_LAZY)) {
    logging::error("Error opening library '%s'", soName.c_str());
    return false;
  }
  const std::string symName("createReducer");
  auto createFn = m_dl.loadSymbol(symName.c_str());
  if (!createFn) {
    logging::error("Symbol '%s' not found in %s",
                   symName.c_str(), soName.c_str());
    return false;
  }
  auto instance = reinterpret_cast<reducerAlgoFactoryFn_t*>(createFn)(m_para, m_pool, det);
  if (!instance)
  {
    logging::error("Error calling %s from %s", symName.c_str(), soName.c_str());
    return false;
  }
  m_algo = instance;

  logging::info("Loaded reducer library '%s' for %u workers", soName.c_str(), m_para.nworkers);
  return true;
}

int Reducer::_setupGraph(unsigned worker)
{
  cudaGraph_t      graph;
  cudaGraphExec_t& graphExec = m_graphExecs[worker];
  cudaStream_t     stream    = m_streams[worker];

  // Build the graph
  logging::debug("Recording Reducer graph %u", worker);
  graph = _recordGraph(worker);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the executable graph
  if (chkError(cudaGraphInstantiate(&graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reducer graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reducer graph %u...", worker);
  if (chkError(cudaGraphUpload(graphExec, stream), "Reducer graph upload failed")) {
    return -1;
  }

  return 0;
}

/** This kernel receives a message from TebReceiver that indicates which
 * calibBuffer is ready for reducing.
 */
static __global__
void _reducerRcv(unsigned*                const __restrict__ state,
                 unsigned*                const __restrict__ index,
                 unsigned*                const __restrict__ retCode,
                 RingQueueHtoD<unsigned>* const __restrict__ inputQueue,
                 uint64_t*                const __restrict__ stateMon,
                 uint64_t* const                __restrict__ inpWtCtr,
                 StageTime*               const __restrict__ time,
                 StageHist*               const __restrict__ hist)
{
  if (*state == 0) {
    //*stateMon = 1;
    //printf("### reducerRcv: wait for idx\n");
    unsigned ns{8};
    while (!inputQueue->pop(index)) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 2;
        return;
      }
    }
    //printf("### _reducerRcv: got idx %u\n", *index);
    *retCode = 0;
    stageTimeStart(time);
    if (hist)  stageHistStart(hist);
    *state = 1;
    //*stateMon = 3;
    //++(*inpWtCtr);
  }
}

/** This will re-launch the current graph */
static __global__
void _reducerLoop(unsigned*                    const __restrict__ state,
                  unsigned const*              const __restrict__ index,
                  unsigned const*              const __restrict__ retCode,
                  uint8_t*                     const __restrict__ dataBuffers,
                  size_t                       const              dataBufsCnt,
                  size_t                       const              rawSize,
                  RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
                  uint64_t*                    const __restrict__ stateMon,
                  uint64_t*                    const __restrict__ outWtCtr,
                  StageTime*                   const __restrict__ time,
                  StageHist*                   const __restrict__ hist,
                  cuda::std::atomic<unsigned>  const&             terminate)
{
  if (*state == 2) {
    if (hist)  stageHistWorkDone(hist);
    //*stateMon = 4;
    auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
    // The size the algorithm recorded, in the word just below whatever it wrote:
    // below the payload normally, and below the raw block when the Detector asked
    // for one, since the raw block occupies the space the size would otherwise use.
    // Either way the slot is in the header reserve or the raw region, both of which
    // are read here before the recorder copies the Dgram over them.
    auto dataSize = ((size_t*)(data - rawSize))[-1];
    //printf("### _reducerLoop: pushing {%u, %lu}\n", *index, dataSize);
    bool rc;
    unsigned ns{8};
    while ( (rc = !outputQueue->push({*index, dataSize, *retCode})) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 5;
        break;
      }
    }
    if (rc) {
      if (hist)  ++hist->pushFulls;
    }
    else {
      //printf("### _reducerLoop: pushed {%u, %lu}\n", *index, dataSize);
      stageTimePushed(time);
      if (hist)  stageHistPushed(hist);
      *state = 0;
      //*stateMon = 6;
      //++(*outWtCtr);
    }
  }

  // This will re-launch the current graph
  //printf("### _reducerLoop: relaunch\n");
  if (!terminate.load(cuda::std::memory_order_acquire))  {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
  //else {
  //  printf("### _reducerLoop: Terminate is True: not relaunching\n");
  //}
}

cudaGraph_t Reducer::_recordGraph(unsigned worker)
{
  red_scoped_range r{/*"Reducer::_recordGraph"*/}; // Expose function name via NVTX

  auto stream       = m_streams[worker];
  auto calibBuffers = m_pool.calibBuffers_d();
  auto calibBufsSz  = m_pool.calibBufsSize();
  auto calibBufsCnt = calibBufsSz / sizeof(*calibBuffers);
  auto dataBuffers  = m_pool.reduceBuffers_d();
  auto dataBufsRsvd = m_pool.reduceBufsReserved();
  auto dataBufsSz   = m_pool.reduceBufsSize();
  // The stride is every region of a reduce buffer, not just the reserve plus
  // payload, so that `&dataBuffers[idx * dataBufsCnt]` indexes buffer idx
  auto dataBufsCnt  = m_pool.reduceBufsStride() / sizeof(*dataBuffers);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Reducer stream begin capture failed")) {
    return 0;
  }

  // Handle messages from TebReceiver to process an event
  _reducerRcv<<<1, 1, 0, stream>>>(m_state_d[worker],
                                   m_indices_d[worker],
                                   m_retCode_d[worker],
                                   m_inputQueues[worker].d,
                                   m_metrics.state[worker],
                                   m_metrics.inpWtCtr[worker],
                                   m_time_d[worker],
                                   m_hist_d[worker]);
  chkError(cudaGetLastError(), "Launch of _reducerRcv kernel failed");

  // Perform the reduction algorithm
  m_algo->recordGraph(stream,
                      m_state_d[worker],
                      m_indices_d[worker],
                      calibBuffers,
                      calibBufsCnt,
                      dataBuffers,
                      dataBufsCnt,
                      m_retCode_d[worker]);

  // Post the completed buffer results and relaunch
  _reducerLoop<<<1, 1, 0, stream>>>(m_state_d[worker],
                                    m_indices_d[worker],
                                    m_retCode_d[worker],
                                    dataBuffers,
                                    dataBufsCnt,
                                    m_pool.reduceBufsRaw(),
                                    m_outputQueues[worker].d,
                                    m_metrics.state[worker],
                                    m_metrics.outWtCtr[worker],
                                    m_time_d[worker],
                                    m_hist_d[worker],
                                    m_terminate_d);
  chkError(cudaGetLastError(), "Launch of _reducerLoop kernel failed");

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Reducer stream end capture failed")) {
    return 0;
  }

  return graph;
}

int Reducer::configure(const json& configureMsg,
                       const json& connectMsg,
                       size_t      collectionId) // Called during phase 1 of Configure
{
  // Configure algorithm
  if (m_algo) {
    if (m_algo->configure(configureMsg, connectMsg, collectionId)) {
      logging::error("Failed 1st configure of Reducer");
      return -1;
    }
  }
  return 0;
}

bool Reducer::setup(Xtc& xtc, const void* bufEnd) // Called during phase 1 of Configure
{
  if (m_algo) {
    // Configure algorithm instances
    if (m_algo->configure(xtc, bufEnd)) {
      logging::error("Failed 2nd configure of Reducer");
      return true;
    }
    // Prepare the CUDA graphs
    m_graphExecs.resize(m_para.nworkers);
    for (unsigned i = 0; i < m_para.nworkers; ++i) {
      if (_setupGraph(i)) {
        logging::error("Failed to set up Reducer graph[%u]", i);
        return true;
      }
    }
  }
  return false;
}

bool Reducer::startup()                 // Called during phase 1 of Configure
{
  logging::info("Starting %u Reducer(s)", m_para.nworkers);

  // Launch the Reducer graphs, which then relaunch themselves until terminated
  for (unsigned i = 0; i < m_graphExecs.size(); ++i) {
    chkFatal(cudaGraphLaunch(m_graphExecs[i], m_streams[i]));
  }

  return false;
}

void Reducer::dump() const
{
  for (unsigned i = 0; i < m_inputQueues.size(); ++i) {
    printf("Reducer %u: in: head %u, tail %u, out: head %u, tail %u\n", i,
           m_inputQueues[i].h->head(), m_inputQueues[i].h->tail(),
           m_outputQueues[i].h->head(), m_outputQueues[i].h->tail());
  }
}

void Reducer::reportHistograms() const
{
  for (unsigned i = 0; i < m_hist_d.size(); ++i) {
    std::string name{"Reducer[" + std::to_string(i) + "]"};
    reportStageHist(name.c_str(), m_hist_d[i]);
  }
}
