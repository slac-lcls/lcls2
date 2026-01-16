#ifndef RINGINDEX_DTOD_HH
#define RINGINDEX_DTOD_HH

#include "psalg/utils/SysLog.hh"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/atomic>

#ifndef __NVCC__
#define __nanosleep(x) {}
#endif

namespace Drp {
  namespace Gpu {

class RingIndexDtoD
{
public:
  __host__ RingIndexDtoD(const unsigned                     capacity,
                         const cuda::std::atomic<unsigned>& terminate_d) :
    m_head_h      (nullptr),
    m_head_d      (nullptr),
    m_tail_h      (nullptr),
    m_tail_d      (nullptr),
    m_capacityMask(capacity-1),    // Range of the buffer index [0, capacity-1]
    m_terminate_d (terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndexDtoD capacity must be a power of 2, got %d\n", capacity);
      abort();
    }

    // Head points to the next index to be allocated
    chkError(cudaHostAlloc(&m_head_h, sizeof(*m_head_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_head_d, m_head_h, 0));
    *m_head_h = 0;
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;
  }

  __host__ ~RingIndexDtoD()
  {
    printf("*** RingQueueDtoD::dtor\n");
    if (m_tail_h)  chkError(cudaFreeHost(m_tail_h));
    if (m_head_h)  chkError(cudaFreeHost(m_head_h));
  }

  __device__ unsigned allocate()                         // Return next index in monotonic fashion
  {
    auto idx  = m_head_d->load(cuda::memory_order_acquire);
    auto next = (idx+1) & m_capacityMask;
    unsigned ns = 8;
    while (next == m_tail_d->load(cuda::memory_order_acquire)) { // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::std::memory_order_acquire))  break;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
    }
    return idx;                                          // Caller now fills buffer[idx]
  }

  __device__ void post(unsigned idx)                     // Move head forward
  {
    unsigned ns = 8;
    while (idx != m_head_d->load(cuda::memory_order_acquire)) {
      if (m_terminate_d.load(cuda::std::memory_order_acquire))  break;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
    }
    auto next = (idx+1) & m_capacityMask;
    m_head_d->store(next, cuda::memory_order_release);   // Publish new head
  }

  __device__ unsigned pend() const                       // Return current head
  {
    auto tail = m_tail_d->load(cuda::memory_order_acquire);
    auto head = m_head_d->load(cuda::memory_order_acquire);
    unsigned ns = 8;
    while (tail == head) {                               // Wait for head to advance while empty
      if (m_terminate_d.load(cuda::std::memory_order_acquire))  break;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      head = m_head_d->load(cuda::memory_order_acquire); // Refresh head
    }
    return head;                                         // Caller now processes buffers up to [head]
  }

  __host__ void release(const unsigned idx)              // Move tail forward
  {
    assert(idx == m_tail_h->load(cuda::memory_order_acquire)); // Require in-order freeing
    auto next = (idx+1) & m_capacityMask;
    m_tail_h->store(next, cuda::memory_order_release);   // Publish new tail
  }

  __host__ unsigned occupancy() const
  {
    return (m_tail_h->load(cuda::memory_order_acquire) -
            m_head_h->load(cuda::memory_order_acquire)) & m_capacityMask;
  }

private:
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head_h; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head_d; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_h; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_d; // Must stay coherent across streams
  const unsigned                                     m_capacityMask;
  const cuda::std::atomic<unsigned>&                 m_terminate_d;
};

  }
}

#endif // RINGINDEX_DTOD_H
