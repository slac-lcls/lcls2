#include "Reducer.hh"

#include "Detector.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "ReducerAlgo.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp;
using namespace Drp::Gpu;
using namespace Pds::Eb;


Reducer::Reducer(const Parameters&        para,
                 MemPoolGpu&              pool,
                 Detector&                det,
                 const std::atomic<bool>& terminate_h,
                 const cuda::atomic<int>& terminate_d) :
  m_pool       (pool),
  m_algo       (nullptr),
  m_terminate_h(terminate_h),
  m_terminate_d(terminate_d),
  m_para       (para)
{
  // Set up buffer index queue for Host to Reducer comms
  m_reducerQueue.h = new Gpu::RingIndexHtoD(m_pool.nbuffers(), m_terminate_h, m_terminate_d);
  chkError(cudaMalloc(&m_reducerQueue.d,                   sizeof(*m_reducerQueue.d)));
  chkError(cudaMemcpy( m_reducerQueue.d, m_reducerQueue.h, sizeof(*m_reducerQueue.d), cudaMemcpyHostToDevice));

  // Set up buffer index queue for Reducer to Host comms
  m_outputQueue.h = new Gpu::RingIndexDtoH(m_pool.nbuffers(), m_terminate_h, m_terminate_d);
  chkError(cudaMalloc(&m_outputQueue.d,                  sizeof(*m_outputQueue.d)));
  chkError(cudaMemcpy( m_outputQueue.d, m_outputQueue.h, sizeof(*m_outputQueue.d), cudaMemcpyHostToDevice));

  // Set up a done flag to cache m_terminate's value and avoid some PCIe transactions
  chkError(cudaMalloc(&m_done,    sizeof(*m_done)));
  chkError(cudaMemset( m_done, 0, sizeof(*m_done)));
  //printf("*** Reducer: m_done %p\n", m_done);

  // Create the Reducer streams
  m_streams.resize(m_para.nworkers);
  //m_events.resize(m_para.nworkers);
  m_heads.resize(m_para.nworkers);
  m_tails.resize(m_para.nworkers);
  for (unsigned i = 0; i < m_streams.size(); ++i) {
    chkFatal(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
    //chkError(cudaEventCreateWithFlags(&m_events[i], cudaEventDisableTiming));

    // Keep track of the head and tail indices of the Reducer stream
    chkError(cudaMalloc(&m_heads[i],    sizeof(*m_heads[i])));
    chkError(cudaMemset( m_heads[i], 0, sizeof(*m_heads[i])));
    chkError(cudaMalloc(&m_tails[i],    sizeof(*m_tails[i])));
    chkError(cudaMemset( m_tails[i], 0, sizeof(*m_tails[i])));
  }
  logging::debug("Done with creating %u Reducer streams", m_streams.size());

  // The header consists of the Dgram with the parent Xtc, the ShapesData Xtc, the
  // Shapes Xtc with its payload and Data Xtc, the payload of which is on the GPU.
  auto headerSize  = sizeof(Dgram) + 3 * sizeof(Xtc) + MaxRank * sizeof(uint32_t);
  auto payloadSize = m_pool.calibBufSize();
  auto totalSize   = headerSize + payloadSize;
  if (totalSize < m_para.maxTrSize)  payloadSize = m_para.maxTrSize - headerSize;

  // Prepare buffers to receive the reduced data,
  // prepended with some reserved space for the datagram header.
  // The application sees only the pointer to the data buffer.
  m_pool.createReduceBuffers(payloadSize, headerSize);

  // Set up the reducer algorithm
  m_algo = _setupAlgo(det);
  if (!m_algo) {
    logging::critical("Error setting up Reducer Algorithm");
    abort();
  }

  // Prepare the CUDA graphs
  m_graphExecs.resize(m_streams.size());
  for (unsigned i = 0; i < m_streams.size(); ++i) {
    if (_setupGraph(i)) {
      logging::critical("Failed to set up Reducer graph");
      abort();
    }
  }
}

Reducer::~Reducer()
{
  printf("*** Reducer dtor 1\n");
  for (auto& graphExec : m_graphExecs) {
    chkError(cudaGraphExecDestroy(graphExec));
  }
  printf("*** Reducer dtor 2\n");

  if (m_algo)  delete m_algo;
  m_dl.close();
  printf("*** Reducer dtor 3\n");

  for (unsigned i = 0; i < m_streams.size(); ++i) {
    chkError(cudaFree(m_tails[i]));
    chkError(cudaFree(m_heads[i]));

    chkError(cudaStreamDestroy(m_streams[i]));
  }
  printf("*** Reducer dtor 4\n");

  chkError(cudaFree(m_done));
  printf("*** Reducer dtor 5\n");

  chkError(cudaFree(m_outputQueue.d));
  delete m_outputQueue.h;
  printf("*** Reducer dtor 6\n");

  chkError(cudaFree(m_reducerQueue.d));
  delete m_reducerQueue.h;
  printf("*** Reducer dtor 7\n");
}

ReducerAlgo* Reducer::_setupAlgo(Detector& det)
{
  // @todo: In the future, find out which Reducer to load from the Detector's configDb entry
  //        For now, load it according to a command line kwarg parameter
  std::string reducer;
  if (m_para.kwargs.find("reducer") == m_para.kwargs.end()) {
    logging::error("Missing required kwarg 'reducer'");
    return nullptr;
  }
  reducer = const_cast<Parameters&>(m_para).kwargs["reducer"];

  if (m_algo)  delete m_algo;     // If the object exists, delete it
  m_dl.close();                   // If a lib is open, close it first

  const std::string soName("lib"+reducer+".so");
  logging::debug("Loading library '%s'", soName.c_str());
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
  auto instance = reinterpret_cast<reducerAlgoFactoryFn_t*>(createFn)(m_para, m_pool, det);
  if (!instance)
  {
    logging::error("Error calling %s from %s", symName.c_str(), soName.c_str());
    return nullptr;
  }
  return instance;
}

int Reducer::_setupGraph(unsigned instance)
{
  cudaGraph_t      graph;
  cudaGraphExec_t& graphExec = m_graphExecs[instance];
  cudaStream_t     stream    = m_streams[instance];

  //printf("*** Reducer setupGraph 1.%u\n", instance);
  // Build the graph
  logging::debug("Recording Reducer graph %u", instance);
  graph = _recordGraph(instance);
  //printf("*** Reducer setupGraph 2.%u\n", instance);
  if (graph == 0) {
    return -1;
  }
  //printf("*** Reducer setupGraph 3.%u\n", instance);

  // Instantiate the executable graph
  if (chkError(cudaGraphInstantiate(&graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reducer graph create failed")) {
    return -1;
  }
  //printf("*** Reducer setupGraph 4.%u\n", instance);

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reducer graph %u...", instance);
  if (chkError(cudaGraphUpload(graphExec, stream), "Reducer graph upload failed")) {
    return -1;
  }
  //printf("*** Reducer setupGraph 5.%u\n", instance);

  return 0;
}

//// This kernel receives a message from TebReceiver that indicates which
//// calibBuffer is ready for reducing.  If the corresponding output data
//// buffer is available for filling by the reduction algorithm, the
//// reduction can proceed.  Otherwise the program stalls.
//static __global__ void _receive(unsigned*            __restrict__ head,
//                                unsigned*            __restrict__ tail,
//                                RingIndexHtoD*       __restrict__ inputQueue,
//                                const RingIndexDtoH* __restrict__ outputQueue,
//                                const cuda::atomic<int>&          terminate,
//                                bool*                __restrict__ done)
//{
//  //printf("### _receive 1 done %d, tail %u, head %u\n", *done, *tail, *head);
//
//  // Refresh the head when the tail has caught up to it
//  // It might be desireable to refresh the head on every call, but that could
//  // prevent progressing the tail toward the head since it blocks when there
//  // is no change.  @todo: Revisit this
//  if (*tail == *head) {
//    //printf("### _receive 2\n");
//
//    // Get the next index to process from the TebReceiver message
//    unsigned idx;
//    while ((idx = inputQueue->consume()) == *head) { // This can block
//      if ( (*done = terminate.load(cuda::memory_order_acquire)) )  return;
//    }
//    *head = idx;                     // This may advance by more than one count
//  }
//
//  // Stall if the output buffer at the tail index is not free
//  if (!outputQueue->empty()) {
//    while (*tail == outputQueue->tail()) {
//      if ( (*done = terminate.load(cuda::memory_order_acquire)) )  return;
//    }
//  }
//}

// This will re-launch the current graph
static __global__ void _graphLoop(unsigned*       __restrict__ tail,
                                  unsigned*       __restrict__ head,
                                  uint8_t** const __restrict__ dataBuffers,
                                  unsigned                     extent) //,
                                  //unsigned                     bufferMask,
                                  //cudaEvent_t                  event,
                                  //cudaStream_t                 stream,
                                  //RingIndexDtoH*  __restrict__ outputQueue,
                                  //const bool&                  done)
{
  //printf("### Reducer graphLoop 1, done %d, idx %u\n", done, *index);
  //if (done)  return;
  //printf("### Reducer graphLoop 1a, index %u\n", *index);

  // Store the extent with the data
  auto data = (uint32_t*)(dataBuffers[*tail]);
  data[-1] = extent;            // @todo: Revisit this kludge to set the extent

  // Push index to host
  //*index = outputQueue->produce(*index);
  //printf("### Reducer graphLoop 2, index %u\n", *index);
  //*index = (*index + 1) & bufferMask;
  *head = *tail;

  //// Signal to the host that we've produced
  //cudaEventRecord(event, stream);

  // Commented out to let TebRcvr::complete() launch the graph
  //cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  //printf("### Reducer graphLoop 3\n");
}

cudaGraph_t Reducer::_recordGraph(unsigned instance)
{
  //printf("*** Reducer::record 1\n");
  auto stream       = m_streams[instance];
  //auto event        = m_events[instance];
  auto calibBuffers = m_pool.calibBuffers_d();
  auto dataBuffers  = m_pool.reduceBuffers_d();
  //auto bufferMask   = m_pool.nbuffers()-1;

  //printf("*** Reducer::record 2\n");
  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Reducer stream begin capture failed")) {
    return 0;
  }
  //printf("*** Reducer::record 3, head %p, tail %p\n", m_heads[instance], m_tails[instance]);

  //// Handle messages from TebReceiver to process an event
  //_receive<<<1, 1, 0, stream>>>(m_heads[instance],
  //                              m_tails[instance],
  //                              m_reducerQueue.d,
  //                              m_outputQueue.d,
  //                              m_terminate_d,
  //                              m_done);
  ////printf("*** Reducer::record 4, algo %p\n", m_algo);

  // Perform the reduction algorithm
  unsigned extent;
  m_algo->recordGraph(stream, *m_tails[instance], calibBuffers, dataBuffers, &extent);
  //printf("*** Reducer::record 5\n");

  // Re-launch! Additional behavior can be put in graphLoop as needed.
  _graphLoop<<<1, 1, 0, stream>>>(m_tails[instance],
                                  m_heads[instance],
                                  dataBuffers,
                                  extent); //,
                                  //bufferMask,
                                  //event,
                                  //stream,
                                  //m_outputQueue.d,
                                  //*m_done);
  //printf("*** Reducer::record 6\n");

  // Signal to the host that the worker is done
  //chkError(cudaEventRecord(event, stream));

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Reducer stream end capture failed")) {
    return 0;
  }

  return graph;
}

void Reducer::start(unsigned worker, unsigned index)
{
  auto  instance  = worker % m_para.nworkers;
  auto& head      = m_heads[instance];
  auto& tail      = m_tails[instance];
  auto& stream    = m_streams[instance];
  auto& graphExec = m_graphExecs[instance];

  //printf("*** Reducer[%d] starting", instance);

  // @todo: Can we arrange for this be done only once in the thread?
  //chkError(cuCtxSetCurrent(m_pool.context().context()));  // Not needed?

  //printf("*** Reducer::start[%u]: 1 idx %u\n", instance, index);
  // Indicate which buffer to reduce
  //m_reducerQueue.h->produce(index);
  //printf("*** Reducer::start[%u]: 2\n", instance);

  // Wait for the graph to finish executing before updating tail
  unsigned h, t;
  do {
    chkError(cudaMemcpyAsync((void*)&h, head, sizeof(*head), cudaMemcpyDeviceToHost, stream));
    chkError(cudaMemcpyAsync((void*)&t, tail, sizeof(*tail), cudaMemcpyDeviceToHost, stream));
    chkError(cudaStreamSynchronize(stream));
    //printf("*** Reducer::start[%u]: tail %d, head %d\n", instance, t, h);
  } while (h != t);                     // Wait if the kernel is still processing
  chkError(cudaMemcpyAsync((void*)tail, &index, sizeof(index), cudaMemcpyHostToDevice, stream));

  // Launch the Reducer graph
  chkFatal(cudaGraphLaunch(graphExec, stream));

  //printf("*** Reducer::start[%u]: 3\n", instance);
}

unsigned Reducer::receive(unsigned worker)
{
  auto  instance = worker % m_para.nworkers;
  //auto& head     = m_heads[instance];
  auto& stream   = m_streams[instance];
  //auto& event    = m_events[instance];
  //printf("*** Reducer::receive[%u]: 1\n", instance);

  //auto index = m_outputQueue.h->consume();
  //printf("*** Reducer::receive[%u]: 2 idx %u\n", instance, index);

  // Wait for the graph to produce the next index
  //chkError(cudaStreamWaitEvent(stream, event, 0));
  //chkError(cudaEventSynchronize(event));

  //unsigned index;
  //chkError(cudaMemcpyAsync((void*)&index, head, sizeof(*head), cudaMemcpyDeviceToHost, stream));
  chkError(cudaStreamSynchronize(stream));

  //return index;
  return 0;
}
