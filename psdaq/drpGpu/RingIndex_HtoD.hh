#ifndef RINGINDEX_HTOD_H
#define RINGINDEX_HTOD_H

#include "psalg/utils/SysLog.hh"

#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace Drp {
  namespace Gpu {

class RingIndexHtoD
{
public:
  __host__ RingIndexHtoD(unsigned                 capacity,
                         const std::atomic<bool>& terminate_h,
                         const cuda::atomic<int>& terminate_d) :
    m_capacity   (capacity),            // Range of the buffer index [0, capacity-1]
    m_terminate_h(terminate_h),
    m_terminate_d(terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndexHtoD capacity must be a power of 2, got %d\n", capacity);
      abort();
    }

    // The ring is empty when head == tail
    chkError(cudaMallocManaged(&m_head, sizeof(*m_head))); // Points to the next index to be allocated
    *m_head = 0;
    chkError(cudaMallocManaged(&m_tail, sizeof(*m_tail))); // Points to the next index after the last freed one
    *m_tail = 0;
  }

  __host__ ~RingIndexHtoD()
  {
    chkError(cudaFree(m_head));
    chkError(cudaFree(m_tail));
  }

  __host__ unsigned produce(unsigned idx)                // Move  head forward
  {
    //printf("***   HtoD rb::produce 1, idx %d, head %d\n", idx, m_head->load());
    auto next = (idx+1)&(m_capacity-1);
    auto tail = m_tail->load(cuda::memory_order_acquire);
    //printf("***   HtoD rb::produce 2, nxt %d, tail %d\n", next, tail);
    while (next == tail) {                               // Wait for tail to advance while full
      if (m_terminate_h.load(std::memory_order_acquire))
        break;
      tail = m_tail->load(cuda::memory_order_acquire);   // Refresh tail
    }
    //printf("***   HtoD rb::produce 3, tail %d\n", tail);
    m_head->store(next, cuda::memory_order_release);     // Publish new head
    //printf("***   HtoD rb::produce 4, head %d\n", m_head->load());
    return next;
  }

  __device__ unsigned consume()                          // Return current head
  {
    //printf("###   HtoD rb::consume 1\n");
    auto tail = m_tail->load(cuda::memory_order_acquire);
    auto head = m_head->load(cuda::memory_order_acquire);
    //printf("###   HtoD rb::consume 2, tail %d, head %d\n", tail, head);
    while (tail == head) {                               // Wait for head to advance while empty
      //printf("###   HtoD rb::consume idx %d\n", head);
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
      head = m_head->load(cuda::memory_order_acquire);   // Refresh head
    }
    //printf("###   HtoD rb::consume 3, idx %d\n", head);
    return head;                                         // Caller now processes buffers up to [head]
  }

  __device__ void release(const unsigned idx)            // Move tail forward
  {
    //printf("###   HtoD rb::release 1, idx %d\n", idx);
    assert(idx == m_tail->load(cuda::memory_order_acquire)); // Require in-order freeing
    auto next = (idx+1)&(m_capacity-1);
    //printf("###   HtoD rb::release 2, nxt %d, cap %d\n", next, m_capacity);
    m_tail->store(next, cuda::memory_order_release);     // Publish new tail
    //printf("###   HtoD rb::release 3, tail %d\n", m_tail->load());
  }

  size_t size()
  {
    return m_capacity;
  }

  void reset()
  {
    *m_head = 0;
    *m_tail = 0;
  }

private:
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_head;   // Must stay coherent across device and host
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_tail;   // Must stay coherent across device and host
  const unsigned           m_capacity;
  const std::atomic<bool>& m_terminate_h;
  const cuda::atomic<int>& m_terminate_d;
};

  } // Gpu
} // Drp

#endif // RINGINDEX_HTOD_H
