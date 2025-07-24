#include "Reducer.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/ResultDgram.hh"
#include "ReducerAlgo.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp;
using namespace Drp::Gpu;
using namespace Pds::Eb;


Reducer::Reducer(unsigned                 instance,
                 const Parameters&        para,
                 MemPoolGpu&              pool,
                 const std::atomic<bool>& terminate_h,
                 const cuda::atomic<int>& terminate_d) :
  m_pool       (pool),
  m_algo       (nullptr),
  m_terminate_h(terminate_h),
  m_terminate_d(terminate_d),
  m_graph      (0),
  m_instance   (instance),
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

  /** Create the Reducer stream **/
  chkFatal(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
  logging::debug("Done with creating Reducer[%u] stream", instance);

  // Set up a done flag to cache m_terminate's value and avoid some PCIe transactions
  chkError(cudaMalloc(&m_done,    sizeof(*m_done)));
  chkError(cudaMemset( m_done, 0, sizeof(*m_done)));
  printf("*** Reducer: m_done %p\n", m_done);

  // Keep track of the head and tail indices of the Reducer stream
  chkError(cudaMalloc(&m_head,    sizeof(*m_head)));
  chkError(cudaMemset( m_head, 0, sizeof(*m_head)));
  chkError(cudaMalloc(&m_tail,    sizeof(*m_tail)));
  chkError(cudaMemset( m_tail, 0, sizeof(*m_tail)));
  printf("*** Reducer: m_head %p, m_tail %p\n", m_head, m_tail);

  // Prepare buffers to receive the reduced data,
  // prepended with some reserved space for the datagram header
  // The application only sees the pointer to the data buffer
  // The header consists of the Dgram with the parent Xtc, the ShapesData Xtc, the
  // Shapes Xtc with its payload and Data Xtc, the payload of which is on the GPU
  // @todo: Get the header size from Gpu::Detector?
  size_t headerSize = sizeof(Dgram) + 3 * sizeof(Xtc) + MaxRank * sizeof(uint32_t);
  m_pool.createReduceBuffers(m_pool.calibBufSize(), headerSize);

  // Set up the reducer algorithm
  m_algo = _setupAlgo();
  if (!m_algo) {
    logging::critical("Error setting up Reducer Algorithm");
    abort();
  }

  // Prepare the CUDA graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Reducer[%u] graph", instance);
    abort();
  }
}

Reducer::~Reducer()
{
  printf("*** Reducer dtor 1\n");
  chkError(cudaGraphExecDestroy(m_graphExec));
  printf("*** Reducer dtor 2\n");
  chkError(cudaGraphDestroy(m_graph)); // @todo: Goes away?
  printf("*** Reducer dtor 3\n");

  printf("*** Reducer dtor 4\n");
  if (m_algo)  delete m_algo;
  m_dl.close();
  printf("*** Reducer dtor 5\n");

  chkError(cudaFree(m_tail));
  printf("*** Reducer dtor 5a\n");
  chkError(cudaFree(m_head));
  printf("*** Reducer dtor 5b\n");
  chkError(cudaFree(m_done));
  printf("*** Reducer dtor 6\n");

  chkError(cudaStreamDestroy(m_stream));
  printf("*** Reducer dtor 7\n");

  chkError(cudaFree(m_outputQueue.d));
  delete m_outputQueue.h;
  printf("*** Reducer dtor 8\n");

  chkError(cudaFree(m_reducerQueue.d));
  delete m_reducerQueue.h;
  printf("*** Reducer dtor 9\n");
}

ReducerAlgo* Reducer::_setupAlgo()
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
  logging::info("Created Reducer from library %s", soName.c_str());
  return instance;
}

int Reducer::_setupGraph()
{
  printf("*** Reducer setupGraph 1\n");
  // Build the graph
  if (m_graph == 0) {        // @todo: Graphs can be created on the stack
    printf("*** Reducer setupGraph 2\n");
    logging::debug("Recording Reducer[%u] graph", m_instance);
    m_graph = _recordGraph(m_stream);
    if (m_graph == 0)
      return -1;
  }
  printf("*** Reducer setupGraph 3\n");

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExec, m_graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Reducer graph create failed")) {
    return -1;
  }
  printf("*** Reducer setupGraph 4\n");

  // @todo: No need to hang on to the stream info
  //cudaGraphDestroy(m_graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Reducer[%u] graph...", m_instance);
  if (chkError(cudaGraphUpload(m_graphExec, m_stream), "Reducer graph upload failed")) {
    return -1;
  }
  printf("*** Reducer setupGraph 5\n");

  return 0;
}

// This kernel receives a message from TebReceiver that indicates which
// calibBuffer is ready for reducing.  If the corresponding output data
// buffer is available for filling by the reduction algorithm, the
// reduction can proceed.  Otherwise the program stalls.
static __global__ void _receive(unsigned*            __restrict__ head,
                                unsigned*            __restrict__ tail,
                                RingIndexHtoD*       __restrict__ inputQueue,
                                const RingIndexDtoH* __restrict__ outputQueue,
                                const cuda::atomic<int>&          terminate,
                                bool*                __restrict__ done)
{
  printf("*** _receive 1 tail %u, head %u\n", *tail, *head);

  // Refresh the head when the tail has caught up to it
  // It might be desireable to refresh the head on every call, but that could
  // prevent progressing the tail toward the head since it blocks when there
  // is no change.  @todo: Revisit this
  if (*tail == *head) {
    printf("*** _receive 2\n");

    // Get the next index to process from the TebReceiver message
    unsigned idx;
    while ((idx = inputQueue->consume()) == *head) { // This can block
      if ( (*done = terminate.load(cuda::memory_order_acquire)) )  return;
    }
    *head = idx;                     // This may advance by more than one count
  }

  // Stall if the output buffer at the tail index is not free
  if (!outputQueue->empty()) {
    while (*tail == outputQueue->tail()) {
      if ( (*done = terminate.load(cuda::memory_order_acquire)) )  return;
    }
  }
}

// This will re-launch the current graph
static __global__ void _graphLoop(unsigned*       __restrict__ index,
                                  uint8_t** const __restrict__ dataBuffers,
                                  unsigned                     extent,
                                  RingIndexDtoH*  __restrict__ outputQueue,
                                  const bool&                  done)
{
  printf("*** Reducer graphLoop 1\n");
  if (done)  return;
  printf("*** Reducer graphLoop 1a, index %u\n", *index);

  // Store the extent with the data
  auto data = (uint32_t*)(dataBuffers[*index]);
  data[-1] = extent;            // @todo: Revisit this kludge to set the extent

  // Push index to host
  *index = outputQueue->produce(*index);
  printf("*** Reducer graphLoop 2, index %u\n", *index);

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  printf("*** Reducer graphLoop 3\n");
}

cudaGraph_t Reducer::_recordGraph(cudaStream_t& stream)
{
  printf("*** Reducer::record 1\n");
  auto calibBuffers = m_pool.calibBuffers_d();
  auto dataBuffers  = m_pool.reduceBuffers_d();

  printf("*** Reducer::record 2\n");
  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Reducer stream begin capture failed")) {
    return 0;
  }
  printf("*** Reducer::record 3, head %p, tail %p\n", m_head, m_tail);

  // Handle messages from TebReceiver to process an event
  _receive<<<1, 1, 0, stream>>>(m_head,
                                m_tail,
                                m_reducerQueue.d,
                                m_outputQueue.d,
                                m_terminate_d,
                                m_done);
  printf("*** Reducer::record 4, algo %p\n", m_algo);

  // Perform the reduction algorithm
  unsigned extent;
  m_algo->recordGraph(stream, *m_tail, calibBuffers, dataBuffers, &extent);
  printf("*** Reducer::record 5\n");

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  _graphLoop<<<1, 1, 0, stream>>>(m_tail, dataBuffers, extent, m_outputQueue.d, *m_done);
  printf("*** Reducer::record 6\n");

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Reducer stream end capture failed")) {
    return 0;
  }

  return graph;
}

void Reducer::start()
{
  logging::info("Reducer[%d] starting", m_instance);
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  // Launch the Reducer graph
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));

  printf("*** Reducer[%d] started\n", m_instance);
}
