#ifndef RINGINDEX_DTOH_H
#define RINGINDEX_DTOH_H

#include "psalg/utils/SysLog.hh"

#include <time.h>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>


namespace Drp {
  namespace Gpu {

class RingIndexDtoH
{
public:
  __host__ RingIndexDtoH(unsigned                           capacity,
                         const std::atomic<bool>&           terminate,
                         const cuda::std::atomic<unsigned>& terminate_d) :
    m_capacityMask(capacity-1),    // Range of the buffer index [0, capacity-1]
    m_terminate  (terminate),
    m_terminate_d(terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndexDtoH capacity must be a power of 2, got %d\n", capacity);
      abort();
    }

    // The ring is empty when head == tail
    // Head points to the next index to be allocated
    chkError(cudaHostAlloc(&m_head_h, sizeof(*m_head_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_head_d, m_head_h, 0));
    *m_head_h = 0;
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;
  }

  __host__ ~RingIndexDtoH()
  {
    chkError(cudaFreeHost(m_head_h));
    chkError(cudaFreeHost(m_tail_h));
  }

  __device__ unsigned post(unsigned idx)                      // Move  head forward
  {
    auto next = (idx+1) & m_capacityMask;
    unsigned ns = 8;
    while (next == m_tail_d->load(cuda::std::memory_order_acquire)) { // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::std::memory_order_acquire))  break;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
    }
    m_head_d->store(next, cuda::std::memory_order_release);   // Publish new head
    return next;
  }

  __host__ unsigned pend() const                              // Return current head
  {
    auto tail = m_tail_h->load(cuda::std::memory_order_acquire);
    auto head = m_head_h->load(cuda::std::memory_order_acquire);
    unsigned ns = 8;
    while (tail == head) {                                    // Wait for head to advance while empty
      if (m_terminate.load(std::memory_order_acquire))  break;
      _nsSleep(ns);
      if (ns < 256)  ns *= 2;
      head = m_head_h->load(cuda::std::memory_order_acquire); // Refresh head
    }
    return head;                                              // Caller now processes buffers up to [head]
  }

  __host__ void release(const unsigned idx)                   // Move tail forward
  {
    assert(idx == m_tail_h->load(cuda::std::memory_order_acquire)); // Require in-order freeing
    auto next = (idx+1) & m_capacityMask;
    m_tail_h->store(next, cuda::std::memory_order_release);   // Publish new tail
  }

  __host__ unsigned occupancy() const
  {
    return (m_tail_h->load(cuda::memory_order_acquire) -
            m_head_h->load(cuda::memory_order_acquire)) & m_capacityMask;
  }

private:
  __host__ static int _nsSleep(unsigned ns)
  {
    struct timespec ts{0, ns};
    return nanosleep(&ts, nullptr);
  }

private:
  cuda::std::atomic<unsigned>*       m_head_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_tail_h; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_head_d; // Must stay coherent across device and host
  cuda::std::atomic<unsigned>*       m_tail_d; // Must stay coherent across device and host
  const unsigned                     m_capacityMask;
  const std::atomic<bool>&           m_terminate;
  const cuda::std::atomic<unsigned>& m_terminate_d;
};

  } // Gpu
} // Drp

#endif // RINGINDEX_DTOH_H
