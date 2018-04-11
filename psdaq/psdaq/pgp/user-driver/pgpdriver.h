#ifndef PGPDRIVER_H
#define PGPDRIVER_H

#include <stdint.h>
#include <errno.h>
#include <vector>
#include "spscqueue.hh"

#define VERSION          0x0000
#define SCRATCH          0x0004
#define UP_TIME_CNT      0x0008
#define RESOURCES        0x00800000
#define RESET            0x00800004
#define MON_ENABLE       0x00800010
#define MON_HIST_ADDR_LO 0x00800014
#define MON_HIST_ADDR_HI 0x00800018
#define CLIENTS(i)       (0x00800080 + i*0x20)
#define DMA_LANES(i)     (0x00800100 + i*0x20)
#define PGP_LANES(i)     (0x00C00000 + i*0x10000)
#define PGP_TX_SIM       0x00D00000

// relative PgpLane register offsets
#define LOOPBACK      0x0000
#define COUNT_RESET   0x8000
#define TX_CONTROL    0x8080

// relative DmaLane register offsets
#define CLIENT 0x0000
#define BLOCK_SIZE 0x0004
#define MEM_STATUS 0x0014

// relative PgpTxSim register offsets
#define CONTROL 0x100
#define SIZE 0x104

// relative Client register offsets
#define DESC_ADDR_LO 0x0000 
#define DESC_ADDR_HI 0x0004
#define DESC_FIFO_LO 0x0008
#define DESC_FIFO_HI 0x000C
#define FIFO_DEPTH   0x0010

#define MAX_LANES 4
#define MON_BUFFER_SIZE 0x10000
#define RX_BUFFER_SIZE (4*1024)
#define RX_DESC_SIZE 0x1000

#define SPAD_WRITE 0x55441122

// 2048kB
#define HUGE_PAGE_BITS 21
#define HUGE_PAGE_SIZE (1 << HUGE_PAGE_BITS)

#define PGP_VENDOR 0x1a4a

struct DmaBuffer
{
    void* virt;
    uintptr_t phys;
    uint32_t size;
    uint32_t lane;
    uint32_t dest;
};

struct DmaBufferPool
{
    DmaBufferPool(size_t num_entries, size_t entry_size);
    SPSCQueue<DmaBuffer*> buffer_queue;
private:
    std::vector<DmaBuffer> buffers;
};

class AxisG2Device
{
public:
    AxisG2Device(int device_id);
    void init(DmaBufferPool* pool);
    void status();
    void setup_lanes(int lane_mask);
    void loop_test(int lane_mask, int loopb, int size, int op_code, int fifolo);
    DmaBuffer* read();
    void write();
private:
    void print_dma_lane(const char* name, int addr, int offset, int mask);
    void print_pgp_lane(const char* name, int addr, int offset, int mask);
    uint8_t* pci_resource;
    std::vector<DmaBuffer*> fifo;
    void* rx_desc;
    uint32_t read_index;
    DmaBufferPool* pool;
};

static inline uint32_t get_reg32(uint8_t* base, int reg) {
    __asm__ volatile ("" : : : "memory");
    return *((volatile uint32_t*) (base + reg));
}

static inline void set_reg32(uint8_t* base, int reg, uint32_t value) {
    __asm__ volatile ("" : : : "memory");
    *((volatile uint32_t*) (base + reg)) = value;
}

#define check_error(function, message) ({\
    int64_t result = (int64_t) (function);\
	if ((int64_t) result == -1LL) {\
        printf("[ERROR] %s:%d %s():\n Failed to %s: %s\n", \
                __FILE__, __LINE__, __func__, message, strerror(errno));\
		exit(errno);\
    }\
	result;\
})

#endif // PGPDRIVER_H
