#ifndef RINGINDEX_GPU_H
#define RINGINDEX_GPU_H

#include "psalg/utils/SysLog.hh"

#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace Drp {
  namespace Gpu {

class RingIndex
{
public:
  __host__ RingIndex(unsigned           capacity,
                     unsigned           dmaCount,
                     std::atomic<bool>& terminate_h,
                     cuda::atomic<int>& terminate_d) :
    m_capacity   (capacity),            // Range of the buffer index [0, capacity-1]
    m_dmaBufMask (dmaCount-1),
    m_terminate_h(terminate_h),
    m_terminate_d(terminate_d)
  {
    using logging = psalg::SysLog;

    if (capacity & (capacity - 1)) {
      logging::critical("RingIndex capacity must be a power of 2, got %d\n", capacity);
      abort();
    };
    if (dmaCount & (dmaCount - 1)) {
      logging::critical("DmaBuffer count must be a power of 2, got %d\n", dmaCount);
      abort();
    };

    // The ring is empty when head == tail
    chkError(cudaMalloc(&m_cursor,    sizeof(*m_cursor))); // For maintaining index ordering
    chkError(cudaMemset( m_cursor, 0, sizeof(*m_cursor)));
    chkError(cudaMallocManaged(&m_head, sizeof(*m_head))); // Points to the next index to be allocated
    *m_head = 0;
    chkError(cudaMallocManaged(&m_tail, sizeof(*m_tail))); // Points to the next index after the last freed one
    *m_tail = 0;
  }

  __host__ ~RingIndex()
  {
    chkError(cudaFree(m_cursor));
    chkError(cudaFree(m_head));
    chkError(cudaFree(m_tail));
  }

  __device__ unsigned prepare(const unsigned instance)   // Return next index in monotonic fashion
  {
    //printf("*** rb::prepare 1, instance %d\n", instance);
    //printf("*** rb::prepare 1, cursor %d\n", m_cursor->load());
    //printf("*** rb::prepare 1, dmaMsk %08x\n", m_dmaBufMask);
    unsigned idx;
    while (((idx = m_cursor->load(cuda::memory_order_acquire)) & m_dmaBufMask) != instance) { // Await this stream's turn
      //printf("*** rb::prepare 1.%d idx %d\n", instance, idx);
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** rb::prepare 2.%d, idx %d, dmaMsk %08x\n", instance, idx, m_dmaBufMask);
    m_cursor->store((idx+1)&(m_capacity-1), cuda::memory_order_release); // Let next stream go
    //printf("*** rb::prepare 3.%d, cursor %d, tail %d, head %d, ret %d\n", instance, m_cursor->load(), m_tail->load(), m_head->load(), idx);
    return idx;                                          // Caller now fills buffer[idx]
  }

  __device__ void produce(const unsigned idx)            // Move head forward
  {
    //const unsigned instance = idx & m_dmaBufMask;
    //printf("*** rb::produce 1.%d, idx %d, head %d\n", instance, idx, m_head->load());
    // Make sure head is published in event order so make other streams wait if they get here first
    while (idx != m_head->load(cuda::memory_order_acquire)) { // Out-of-turn streams wait here
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** rb::produce 2.%d, idx %d\n", instance, idx);
    auto next = (idx+1)&(m_capacity-1);
    //printf("*** rb::produce 3.%d, nxt %d\n", instance, next);
    while (m_tail->load(cuda::memory_order_acquire) == next) { // Wait for tail to advance while full
      if (m_terminate_d.load(cuda::memory_order_acquire))
        break;
    }
    //printf("*** rb::produce 4.%d, tail %d\n", instance, m_tail->load());
    m_head->store(next, cuda::memory_order_release);      // Publish new head
    //printf("*** rb::produce 5.%d, head %d\n", instance, m_head->load());
  }

  __host__ unsigned consume()                            // Return current head
  {
    //printf("*** rb::consume 1\n");
    auto tail = m_tail->load(cuda::memory_order_acquire);
    //printf("*** rb::consume 2, tail %d, head %d\n", tail, m_head->load(cuda::memory_order_acquire));
    unsigned idx;
    while (tail == (idx = m_head->load(cuda::memory_order_acquire))) { // Wait for head to advance while empty
      //printf("*** rb::consume idx %d\n", idx);
      if (m_terminate_h.load(std::memory_order_acquire))
        break;
    }
    //printf("*** rb::consume 3, idx %d\n", idx);
    return idx;                                          // Caller now processes buffer[idx]
  }

  __host__ void release(const unsigned idx)              // Move tail forward
  {
    //printf("*** rb::release 1, idx %d\n", idx);
    assert(idx == m_tail->load(cuda::memory_order_acquire)); // Require in-order freeing
    //printf("*** rb::release 2, cap %d\n", m_capacity);
    m_tail->store((idx+1)&(m_capacity-1), cuda::memory_order_release); // Publish new tail
    //printf("*** rb::release 3, tail %d\n", m_tail->load());
  }

  size_t size()
  {
    return m_capacity;
  }

  void reset()
  {
    chkError(cudaMemset(m_cursor, 0, sizeof(*m_cursor)));
    *m_head = 0;
    *m_tail = 0;
  }

private:
  // Not sure why nvcc says the __*__ modifiers are "ignored"
  cuda::atomic<unsigned, cuda::thread_scope_device>* m_cursor; // Must stay coherent across streams
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_head;   // Must stay coherent across device and host
  cuda::atomic<unsigned, cuda::thread_scope_system>* m_tail;   // Must stay coherent across device and host
  const unsigned     m_capacity;
  const unsigned     m_dmaBufMask;
  std::atomic<bool>& m_terminate_h;
  cuda::atomic<int>& m_terminate_d;
};

  };
};

#endif // RINGINDEX_GPU_H
