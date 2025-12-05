#ifndef RINGINDEX_DTOD_H
#define RINGINDEX_DTOD_H

#include "psalg/utils/SysLog.hh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#ifndef __NVCC__
#define __nanosleep(x) {}
#endif

namespace Drp {
  namespace Gpu {

class RingIndexDtoD
{
public:
  __host__ RingIndexDtoD(const unsigned               capacity,
                         const unsigned               dmaCount,
                         const cuda::atomic<uint8_t>& terminate_d) :
    m_capacity   (capacity),        // Range of the buffer index [0, capacity-1]
    m_dmaBufMask (dmaCount-1),
    m_terminate_d(terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndexDtoD capacity must be a power of 2, got %d\n", capacity);
      abort();
    }
    if (dmaCount & (dmaCount - 1)) {
      logging::critical("DmaBuffer count must be a power of 2, got %d\n", dmaCount);
      abort();
    }

    // The ring is empty when head == tail
    chkError(cudaMalloc(&m_cursor,    sizeof(*m_cursor)));  // For maintaining index ordering
    chkError(cudaMemset( m_cursor, 0, sizeof(*m_cursor)));
    // Head points to the next index to be allocated
    chkError(cudaMalloc(&m_head_d,    sizeof(*m_head_d)));
    chkError(cudaMemset( m_head_d, 0, sizeof(*m_head_d)));
    // Tail points to the next index after the last freed one
    chkError(cudaHostAlloc(&m_tail_h, sizeof(*m_tail_h), cudaHostAllocDefault));
    chkError(cudaHostGetDevicePointer(&m_tail_d, m_tail_h, 0));
    *m_tail_h = 0;
  }

  __host__ ~RingIndexDtoD()
  {
    chkError(cudaFree(m_cursor));
    chkError(cudaFree(m_head_d));
    chkError(cudaFreeHost(m_tail_h));
  }

  __device__ unsigned prepare()                          // Return next index in monotonic fashion
  {
    auto idx  = m_head_d->load(cuda::memory_order_acquire);
    auto next = (idx+1)&(m_capacity-1);
    auto tail = m_tail_d->load(cuda::memory_order_acquire);
    while (next == tail) {                               // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
      tail = m_tail_d->load(cuda::memory_order_acquire); // Refresh tail
    }
    return idx;                                          // Caller now fills buffer[idx]
  }

  __device__ void produce(unsigned idx)                  // Move head forward
  {
    //const unsigned instance = idx & m_dmaBufMask;
    auto head = m_head_d->load(cuda::memory_order_acquire);
    //printf("###   DtoD rb::produce 1.%d, idx %d, head %d\n", instance, idx, head);
    // Make sure head is published in event order so make other streams wait if they get here first
    while (idx != head) {                                // Out-of-turn streams wait here
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
      //__nanosleep(5000);                                 // Suspend the thread
      head = m_head_d->load(cuda::memory_order_acquire); // Refresh head
    }
    //printf("###   DtoD rb::produce 2.%d, idx %d\n", instance, idx);
    auto next = (idx+1)&(m_capacity-1);
    //printf("###   DtoD rb::produce 3.%d, nxt %d\n", instance, next);
    m_head_d->store(next, cuda::memory_order_release);   // Publish new head
    //printf("###   DtoD rb::produce 4.%d, head %d\n", instance, m_head_d->load());
  }

  __device__ unsigned consume()                          // Return current head
  {
    //printf("###   DtoD rb::consume 1\n");
    auto tail = m_tail_d->load(cuda::memory_order_acquire);
    auto head = m_head_d->load(cuda::memory_order_acquire);
    //printf("###   DtoD rb::consume 2, tail %d, head %d\n", tail, head);
    while (tail == head) {                               // Wait for head to advance while empty
      //printf("###   DtoD rb::consume idx %d\n", head);
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
      //__nanosleep(5000);                                 // Suspend the thread
      head = m_head_d->load(cuda::memory_order_acquire); // Refresh head
    }
    //printf("###   DtoD rb::consume 3, idx %d\n", head);
    return head;                                         // Caller now processes buffers up to [head]
  }

  __host__ void release(const unsigned idx)              // Move tail forward
  {
    //printf("###   DtoD rb::release 1, idx %d\n", idx);
    assert(idx == m_tail_h->load(cuda::memory_order_acquire)); // Require in-order freeing
    auto next = (idx+1)&(m_capacity-1);
    //printf("###   DtoD rb::release 2, nxt %d, cap %d\n", next, m_capacity);
    m_tail_h->store(next, cuda::memory_order_release);   // Publish new tail
    //printf("###   DtoD rb::release 3, tail %d\n", m_tail_h->load());
  }

  __host__ unsigned head() const
  {
    return m_head_d->load(cuda::memory_order_acquire);
  }

  __host__ unsigned tail() const
  {
    return m_tail_h->load(cuda::memory_order_acquire);
  }

  __host__ bool empty() const
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
    chkError(cudaMemset(m_head_d, 0, sizeof(*m_head_d)));
    *m_tail_h = 0;
  }

private:
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_cursor; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head_d; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_h; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail_d; // Must stay coherent across streams
  const unsigned               m_capacity;
  const unsigned               m_dmaBufMask;
  const cuda::atomic<uint8_t>& m_terminate_d;
};

  }
}

#endif // RINGINDEX_DTOD_H
