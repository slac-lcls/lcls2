#ifndef RINGINDEX_DTOD_H
#define RINGINDEX_DTOD_H

#include "psalg/utils/SysLog.hh"

#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace Drp {
  namespace Gpu {

class RingIndexDtoD
{
public:
  __host__ RingIndexDtoD(const unsigned           capacity,
                         const unsigned           dmaCount,
                         const cuda::atomic<int>& terminate) :
    m_capacity  (capacity),        // Range of the buffer index [0, capacity-1]
    m_dmaBufMask(dmaCount-1),
    m_terminate (terminate)
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
    chkError(cudaMalloc(&m_cursor,    sizeof(*m_cursor))); // For maintaining index ordering
    chkError(cudaMemset( m_cursor, 0, sizeof(*m_cursor)));
    chkError(cudaMalloc(&m_head,      sizeof(*m_head)));   // Points to the next index to be allocated
    chkError(cudaMemset( m_head,   0, sizeof(*m_head)));
    chkError(cudaMalloc(&m_tail,      sizeof(*m_tail)));   // Points to the next index after the last freed one
    chkError(cudaMemset( m_tail,   0, sizeof(*m_tail)));
  }

  __host__ ~RingIndexDtoD()
  {
    chkError(cudaFree(m_cursor));
    chkError(cudaFree(m_head));
    chkError(cudaFree(m_tail));
  }

  __device__ unsigned prepare(unsigned instance)         // Return next index in monotonic fashion
  {
    //printf("*** DtoD rb::prepare 1, instance %d\n", instance);
    //printf("*** DtoD rb::prepare 1, cursor %d\n", m_cursor->load());
    //printf("*** DtoD rb::prepare 1, dmaMsk %08x\n", m_dmaBufMask);
    unsigned idx;
    while (((idx = m_cursor->load(cuda::memory_order_acquire)) & m_dmaBufMask) != instance) { // Await this stream's turn
      //printf("*** DtoD rb::prepare 1.%d idx %d\n", instance, idx);
      if (m_terminate.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** DtoD rb::prepare 2.%d, idx %d, dmaMsk %08x\n", instance, idx, m_dmaBufMask);
    m_cursor->store((idx+1)&(m_capacity-1), cuda::memory_order_release); // Let next stream go
    //printf("*** DtoD rb::prepare 3.%d, cursor %d, tail %d, head %d, ret %d\n", instance, m_cursor->load(), m_tail->load(), m_head->load(), idx);
    return idx;                                          // Caller now fills buffer[idx]
  }

  __device__ void produce(unsigned idx)                  // Move head forward
  {
    //const unsigned instance = idx & m_dmaBufMask;
    //printf("*** DtoD rb::produce 1.%d, idx %d, head %d\n", instance, idx, m_head->load());
    // Make sure head is published in event order so make other streams wait if they get here first
    while (idx != m_head->load(cuda::memory_order_acquire)) { // Out-of-turn streams wait here
      if (m_terminate.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** DtoD rb::produce 2.%d, idx %d\n", instance, idx);
    auto next = (idx+1)&(m_capacity-1);
    //printf("*** DtoD rb::produce 3.%d, nxt %d\n", instance, next);
    while (m_tail->load(cuda::memory_order_acquire) == next) { // Wait for tail to advance while full
      if (m_terminate.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** DtoD rb::produce 4.%d, tail %d\n", instance, m_tail->load());
    m_head->store(next, cuda::memory_order_release);     // Publish new head
    //printf("*** DtoD rb::produce 5.%d, head %d\n", instance, m_head->load());
  }

  __device__ unsigned consume()                          // Return current head
  {
    //printf("*** DtoD rb::consume 1\n");
    auto tail = m_tail->load(cuda::memory_order_acquire);
    //printf("*** DtoD rb::consume 2, tail %d, head %d\n", tail, m_head->load(cuda::memory_order_acquire));
    unsigned idx;
    while (tail == (idx = m_head->load(cuda::memory_order_acquire))) { // Wait for head to advance while empty
      //printf("*** DtoD rb::consume idx %d\n", idx);
      if (m_terminate.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** DtoD rb::consume 3, idx %d\n", idx);
    return idx;                                          // Caller now processes buffer[idx]
  }

  //__device__ void release(const unsigned idx)            // Move tail forward
  __device__ void release(const unsigned idx)            // Move tail forward
  {
    //printf("*** DtoD rb::release 1, idx %d\n", idx);
    assert(idx == m_tail->load(cuda::memory_order_acquire)); // Require in-order freeing
    //printf("*** DtoD rb::release 2, cap %d\n", m_capacity);
    m_tail->store((idx+1)&(m_capacity-1), cuda::memory_order_release); // Publish new tail
    //printf("*** DtoD rb::release 3, tail %d\n", m_tail->load());
  }

  size_t size()
  {
    return m_capacity;
  }

  void reset()
  {
    chkError(cudaMemset(m_cursor, 0, sizeof(*m_cursor)));
    chkError(cudaMemset(m_head,   0, sizeof(*m_head)));
    chkError(cudaMemset(m_tail,   0, sizeof(*m_tail)));
  }

private:
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_cursor; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_head;   // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_tail;   // Must stay coherent across streams
  const unsigned           m_capacity;
  const unsigned           m_dmaBufMask;
  const cuda::atomic<int>& m_terminate;
};

  }
}

#endif // RINGINDEX_DTOD_H
