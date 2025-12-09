#ifndef RINGINDEX_DTOH_H
#define RINGINDEX_DTOH_H

#include "psalg/utils/SysLog.hh"

#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/atomic>

#ifndef __NVCC__
#define __nanosleep(x) {}
#endif

namespace Drp {
  namespace Gpu {

class RingIndexDtoH
{
public:
  __host__ RingIndexDtoH(unsigned                           capacity,
                         const std::atomic<bool>&           terminate,
                         const cuda::std::atomic<unsigned>& terminate_d) :
    m_capacity   (capacity),            // Range of the buffer index [0, capacity-1]
    m_terminate  (terminate),
    m_terminate_d(terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndexDtoH capacity must be a power of 2, got %d\n", capacity);
      abort();
    }

    // The ring is empty when head == tail
    chkError(cudaMalloc(&m_cursor,    sizeof(*m_cursor)));  // For maintaining index ordering
    chkError(cudaMemset( m_cursor, 0, sizeof(*m_cursor)));
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
    chkError(cudaFree(m_cursor));
    chkError(cudaFreeHost(m_head_h));
    chkError(cudaFreeHost(m_tail_h));
  }

//  __device__ unsigned prepare(unsigned idx)              // Return next index in monotonic fashion
//  {
//    //if (idx != m_cursor->load(cuda::memory_order_acquire)) {
//    //  printf("Index differs from expected value %u: %u\n", m_cursor->load(), idx);
//    //  return idx;                       // abort(); ???
//    //}
//    //
//    //auto next = (idx+1) & (m_capacity-1);
//    //
//    ////printf("###   DtoH rb::prepare 1, cursor %d\n", m_cursor->load());
//    ////printf("###   DtoH rb::prepare 2, next %d\n", next);
//    //m_cursor->store(next, cuda::memory_order_release);
//    ////printf("###   DtoH rb::prepare 3, cursor %d, ret %d\n", m_cursor->load(), idx);
//    m_head->store(idx, cuda::memory_order_release);      // Advance new head
//    return idx;                                          // Caller now handles buffer[idx]
//  }

  __device__ unsigned produce(unsigned idx)              // Move  head forward
  {
    //printf("###   DtoH rb::produce 1, idx %d, head %d\n", idx, m_head->load());
    auto next = (idx+1)&(m_capacity-1);
    auto tail = m_tail_d->load(cuda::memory_order_acquire);
    //printf("###   DtoH rb::produce 2, nxt %d, tail %d\n", next, tail);
    while (next == tail) {                               // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::std::memory_order_acquire))
        break;
      //__nanosleep(5000);                                 // Suspend the thread
      tail = m_tail_d->load(cuda::memory_order_acquire); // Refresh tail
    }
    //printf("###   DtoH rb::produce 3, tail %d\n", tail);
    m_head_d->store(next, cuda::memory_order_release);   // Publish new head
    //printf("###   DtoH rb::produce 4, head %d\n", m_head->load());
    return next;
  }

  __host__ unsigned consume()                            // Return current head
  {
    //printf("***   DtoH rb::consume 1\n");
    auto tail = m_tail_h->load(cuda::memory_order_acquire);
    auto head = m_head_h->load(cuda::memory_order_acquire);
    //printf("***   DtoH rb::consume 2, tail %d, head %d\n", tail, head);
    while (tail == head) {                               // Wait for head to advance while empty
      //printf("***   DtoH rb::consume idx %d\n", head);
      if (m_terminate.load(std::memory_order_acquire))
        break;
      head = m_head_h->load(cuda::memory_order_acquire); // Refresh head
    }
    //printf("***   DtoH rb::consume 3, idx %d\n", head);
    return head;                                         // Caller now processes buffers up to [head]
  }

  __host__ void release(const unsigned idx)              // Move tail forward
  {
    //printf("***   DtoH rb::release 1, idx %d\n", idx);
    assert(idx == m_tail_h->load(cuda::memory_order_acquire)); // Require in-order freeing
    auto next = (idx+1)&(m_capacity-1);
    //printf("***   DtoH rb::release 2, nxt %d, cap %d\n", next, m_capacity);
    m_tail_h->store(next, cuda::memory_order_release);   // Publish new tail
    //printf("***   DtoH rb::release 3, tail %d\n", m_tail->load());
  }

  __host__ __device__ unsigned head() const
  {
    return m_head_h->load(cuda::memory_order_acquire);
  }

  __host__ __device__ unsigned tail() const
  {
    return m_tail_h->load(cuda::memory_order_acquire);
  }

  __host__ __device__ bool empty() const
  {
    return head() == tail();
  }

  size_t size()
  {
    return m_capacity;
  }

  void reset()
  {
    chkError(cudaMemset(m_cursor, 0, sizeof(*m_cursor)));
    *m_head_h = 0;
    *m_tail_h = 0;
  }

private:
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_cursor; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_head_h; // Must stay coherent across device and host
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_tail_h; // Must stay coherent across device and host
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_head_d; // Must stay coherent across device and host
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_tail_d; // Must stay coherent across device and host
  const unsigned                                     m_capacity;
  const std::atomic<bool>&                           m_terminate;
  const cuda::std::atomic<unsigned>&                 m_terminate_d;
};

  } // Gpu
} // Drp

#endif // RINGINDEX_DTOH_H
