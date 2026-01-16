#ifndef RINGQUEUE_DTOH_HH
#define RINGQUEUE_DTOH_HH

#include "psalg/utils/SysLog.hh"

#include <time.h>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#ifndef __NVCC__
#define __nanosleep(x) {}
#endif

namespace Drp {
  namespace Gpu {

template <typename T>
class RingQueueDtoH
{
public:
  __host__ RingQueueDtoH(const unsigned                     capacity,
                         const std::atomic<bool>&           terminate,
                         const cuda::std::atomic<unsigned>& terminate_d) :
    m_head_h      (nullptr),
    m_head_d      (nullptr),
    m_tail_h      (nullptr),
    m_tail_d      (nullptr),
    m_capacityMask(capacity-1),    // Range of the buffer queue [0, capacity-1]
    m_ringBuffer_h(nullptr),
    m_ringBuffer_d(nullptr),
    m_terminate   (terminate),
    m_terminate_d (terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingQueue capacity must be a power of 2, got %d\n", capacity);
      abort();
    }

    printf("RingQueueDtoH::ctor\n");

    // The ring is empty when head == tail
    // Head points to the next index to be allocated
    chkError(cudaHostAlloc(&m_head_h, sizeof(*m_head_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_head_d, m_head_h, 0));
    *m_head_h = 0;
    printf("RingQueueDtoH::ctor: head h %p, d %p\n", m_head_h, m_head_d);
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;
    printf("RingQueueDtoH::ctor: tail h %p, d %p\n", m_tail_h, m_tail_d);

    // Use pinned memory for the ring buffer for low latency access from both Host and Device
    chkError(cudaHostAlloc(&m_ringBuffer_h, capacity * sizeof(*m_ringBuffer_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_ringBuffer_d, m_ringBuffer_h, 0));
    printf("RingQueueDtoH::ctor: ringBuf h %p, d %p\n", m_ringBuffer_h, m_ringBuffer_d);
  }

  __host__ ~RingQueueDtoH()
  {
    printf("*** RingQueueDtoH::dtor\n");
    if (m_ringBuffer_h)  chkError(cudaFreeHost(m_ringBuffer_h));
    if (m_tail_h)        chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)        chkError(cudaFreeHost(m_head_h));
  }

  __device__ bool push(T value)
  {
    auto head = m_head_d->load(cuda::std::memory_order_acquire);
    //printf("### RingQueueDtoH push 1, head %u, tail %u\n", head, m_tail_d->load());
    auto next = (head+1) & m_capacityMask;
    unsigned ns = 8;
    while (next == m_tail_d->load(cuda::std::memory_order_acquire)) { // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::std::memory_order_acquire))  return false;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
    }
    //printf("### RingQueueDtoH push 2, head %u, tail %u\n", head, m_tail_d->load());
    m_head_d->store(next, cuda::std::memory_order_release);           // Publish new head
    m_ringBuffer_d[head] = value;
    //printf("### RingQueueDtoH push 3, head %u, tail %u\n", m_head_d->load(), m_tail_d->load());
    return true;
  }

  __host__ bool pop(T& value)
  {
    auto tail = m_tail_h->load(cuda::std::memory_order_acquire);
    //printf("*** RingQueueDtoH pop 1, tail %u, head %u\n", tail, m_head_h->load());
    unsigned ns = 8;
    while (tail == m_head_h->load(cuda::std::memory_order_acquire)) { // Wait for head to advance while empty
      if (m_terminate.load(std::memory_order_acquire))  return false;
      _nsSleep(ns);
      if (ns < 256)  ns *= 2;
    }
    //printf("*** RingQueueDtoH pop 2\n");
    auto next = (tail+1) & m_capacityMask;
    m_tail_h->store(next, cuda::std::memory_order_release);           // Publish new tail
    value = m_ringBuffer_h[tail];
    //printf("*** RingQueueDtoH pop 3\n");
    return true;
  }

  __host__ int occupancy() const
  {
    return (m_head_h->load(cuda::std::memory_order_acquire) -
            m_tail_h->load(cuda::std::memory_order_acquire)) & m_capacityMask;
  }

  size_t size() const
  {
    return m_capacityMask + 1;
  }

  __host__ void reset()
  {
    *m_head_h = 0;
    *m_tail_h = 0;
  }

private:
  __host__ static int _nsSleep(unsigned ns)
  {
    struct timespec ts{0, ns};
    return nanosleep(&ts, nullptr);
  }

private:
  cuda::std::atomic<unsigned>*       m_head_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_head_d; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_tail_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_tail_d; // Must stay coherent across device and host
  unsigned                           m_capacityMask;
  T*                                 m_ringBuffer_h;
  T*                                 m_ringBuffer_d;
  const std::atomic<bool>&           m_terminate;
  const cuda::std::atomic<unsigned>& m_terminate_d;
};

  } // Gpu
} // Drp

#endif // RINGQUEUE_HTOD_HH
