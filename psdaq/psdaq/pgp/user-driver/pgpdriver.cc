#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <cassert>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/limits.h>
#include <string>
#include <fstream>

#include "pgpdriver.h"

// translate a virtual address to a physical address
uintptr_t virt_to_phys(void* virt)
{
    long pagesize = sysconf(_SC_PAGESIZE);
    int fd = check_error(open("/proc/self/pagemap", O_RDONLY), "Error openining /proc/self/pagemap");
    // pagemap is an array of pointers for each normal-sized page
    check_error(lseek(fd, (uintptr_t) virt / pagesize * sizeof(uintptr_t), SEEK_SET), "pagemap");
    uintptr_t phy = 0;
    check_error(read(fd, &phy, sizeof(phy)), "converting address");
    close(fd);
    if (!phy) {
        printf("failed to translate virtual address %p to physical address\n", virt);
    }
    // bits 0-54 are the page number
    return (phy & 0x7fffffffffffffULL) * pagesize + ((uintptr_t) virt) % pagesize;
}

void* memory_allocate_dma(size_t size, const char* path)
{
    // round up to multiples of hugepage size
    if (size % HUGE_PAGE_SIZE) {
        size = ((size >> HUGE_PAGE_BITS) + 1) << HUGE_PAGE_BITS;
    }
    int fd = check_error(open(path, O_CREAT | O_RDWR, S_IRWXU),
                         "open hugetlbfs file, check that /mnt/huge is mounted");
    check_error(ftruncate(fd, (off_t) size), "");
    void* virt_addr = (void*) check_error(mmap(NULL, size, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE, fd, 0),
                                          "mmap hugepage");
    check_error(mlock(virt_addr, size), "disable swap for DMA memory");
    close(fd);
    return virt_addr;
}

void enable_dma(const char* pci_addr)
{
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "/sys/bus/pci/devices/%s/config", pci_addr);
    int fd = check_error(open(path, O_RDWR), "open pci config");
    assert(lseek(fd, 4, SEEK_SET) == 4);
    uint16_t dma = 0;
    assert(read(fd, &dma, 2) == 2);
    dma |= 1 << 2;
    assert(lseek(fd, 4, SEEK_SET) == 4);
    assert(write(fd, &dma, 2) == 2);
    check_error(close(fd), "close");
}

uint8_t* map_pci_resource(const char* pci_addr)
{
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "/sys/bus/pci/devices/%s/resource0", pci_addr);
    enable_dma(pci_addr);
    int fd = check_error(open(path, O_RDWR), "open pci resource");
    struct stat stat;
    check_error(fstat(fd, &stat), "stat pci resource");
    return (uint8_t*)check_error(mmap(NULL, stat.st_size, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0), "mmap pci resource");
}

DmaBufferPool::DmaBufferPool(size_t num_entries, size_t entry_size) : buffer_queue(num_entries),
                                                    buffers(num_entries)
{
    if (HUGE_PAGE_SIZE % entry_size) {
        printf("entry_size must to be a divisor of the huge page size %d kB\n", HUGE_PAGE_SIZE/1024);
        exit(-1);
    }
    void* base = memory_allocate_dma(num_entries * entry_size, "/mnt/huge/pgp_dma_buffers");
    for (size_t i=0; i<num_entries; i++) {
        void* virt = (void*) (((uint8_t*)base) + i*entry_size);
        buffers[i].virt = virt;
        buffers[i].phys = virt_to_phys(virt);
        buffer_queue.push(&buffers[i]);
    }
}

long get_id(std::string file_name)
{
    std::ifstream in(file_name);
    std::string line;
    std::getline(in, line);
    return strtol(line.c_str(), NULL, 0);
}

std::string get_pgp_bus_id(int device_id)
{
    std::string base_dir = "/sys/bus/pci/devices/";
    DIR* parent = opendir(base_dir.c_str());
    while (dirent* child = readdir(parent)) {
        long vendor =  get_id(base_dir + child->d_name + "/vendor");
        long device =  get_id(base_dir + child->d_name + "/device");
        if ((vendor == PGP_VENDOR) & (device == device_id)) {
             printf("Found PGP card at: %s\n", child->d_name);
             return std::string(child->d_name);
        }
    }
    printf("Error no PGP card found! Aborting!\n");
    exit(-1);
}

AxisG2Device::AxisG2Device(int device_id)
{
    std::string bus_id = get_pgp_bus_id(device_id);
    pci_resource = map_pci_resource(bus_id.c_str());
}

void AxisG2Device::init(DmaBufferPool* pool)
{
    this->pool = pool;

    // reset pgp card
    set_reg32(pci_resource, RESET, 1);
    set_reg32(pci_resource, COUNT_RESET, 1);
    set_reg32(pci_resource, SCRATCH, SPAD_WRITE);

    // allocate mon buffer and enable monitoring
    void* mon_addr = memory_allocate_dma(MON_BUFFER_SIZE, "/mnt/huge/pgp_mon");
    uintptr_t phys = virt_to_phys(mon_addr);
    set_reg32(pci_resource, MON_HIST_ADDR_LO, phys & 0xFFFFFFFF);
    set_reg32(pci_resource, MON_HIST_ADDR_HI, (phys >> 32) & 0x000000FF);
    set_reg32(pci_resource, MON_ENABLE, 1);

    uint32_t client_base = CLIENTS(0);
    // allocate receive descriptors that are written to by the hardware
    rx_desc = memory_allocate_dma(RX_DESC_SIZE*8, "/mnt/huge/pgp_rx_desc");
    memset(rx_desc, 0, RX_DESC_SIZE*8);
    phys = virt_to_phys(rx_desc);
    set_reg32(pci_resource, client_base + DESC_ADDR_LO, phys & 0xFFFFFFFF);
    set_reg32(pci_resource, client_base + DESC_ADDR_HI, (phys >> 32) & 0x000000FF);
    read_index = 0;

     // get receive buffers from memory pool and give them to the hardware
    fifo.resize(RX_DESC_SIZE);
    for (int i=0; i<RX_DESC_SIZE; i++) {
        pool->buffer_queue.pop(fifo[i]);
        set_reg32(pci_resource, client_base + DESC_FIFO_LO, fifo[i]->phys & 0xFFFFFFFF);
        set_reg32(pci_resource, client_base + DESC_FIFO_HI, (i << 8) | ((fifo[i]->phys >> 32) & 0x000000FF));
    }
}

void AxisG2Device::print_dma_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, DMA_LANES(i) + addr);
        printf(" %8u", (reg >> offset) & mask);
    }
    printf("\n");
}

void AxisG2Device::print_pgp_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, PGP_LANES(i) + 0x8000 + addr);
        printf(" %8u", (reg >> offset) & mask);
    }
    printf("\n");
}

void AxisG2Device::status()
{
    uint32_t version = get_reg32(pci_resource, VERSION);
    uint32_t scratch = get_reg32(pci_resource, SCRATCH);
    uint32_t uptime_count = get_reg32(pci_resource, UP_TIME_CNT);
    uint32_t lanes = get_reg32(pci_resource, RESOURCES) & 0xf;
    char build_string[256];
    for (int i=0; i<64; i++) {
        reinterpret_cast<uint32_t*>(build_string)[i] = get_reg32(pci_resource, 0x0800 + i*4);
    }  

    printf("-- Core Axi Version --\n");
    printf("  firmware version  :  %x\n", version);
    printf("  scratch           :  %x\n", scratch);
    printf("  uptime count      :  %d\n", uptime_count);
    printf("  build string      :  %s\n", build_string);
    
    printf("  lanes             :  %u\n", lanes);
    uint32_t fifo_depth = get_reg32(pci_resource, CLIENTS(0) + FIFO_DEPTH);
    printf("dcountRamAddr  [%u] : 0x%x\n", 0, fifo_depth & 0xffff);
    printf("dcountWriteDesc[%u] : 0x%x\n", 0, fifo_depth >> 16);
    printf("wrIndex        [%u] : 0x%x\n", 0, read_index);

    printf("\n-- dmaLane Registers --\n");
    print_dma_lane("client", CLIENT, 0, 0xf);
    print_dma_lane("blockSize", BLOCK_SIZE, 0, 0xf);
    print_dma_lane("dcountTransfer", FIFO_DEPTH, 0, 0xffff);
    print_dma_lane("blocksFree", MEM_STATUS, 0, 0x3ff);

    print_dma_lane("blocksQueued", MEM_STATUS, 12, 0x3ff);
    print_dma_lane("tready", MEM_STATUS, 25, 1);
    print_dma_lane("wbusy", MEM_STATUS, 26, 1);
    print_dma_lane("wSlaveBusy", MEM_STATUS, 27, 1);
    print_dma_lane("rMasterBusy", MEM_STATUS, 28, 1);
    print_dma_lane("mm2s_err", MEM_STATUS, 29, 1);
    print_dma_lane("s2mm_err", MEM_STATUS,30, 1);
    print_dma_lane("memReady", MEM_STATUS, 31, 1);

    printf("\n-- PgpAxiL Registers --\n");
    print_pgp_lane("phyRxActive", 0x10, 0, 1);
    print_pgp_lane("locLinkRdy", 0x10, 1, 1);
    print_pgp_lane("remLinkRdy", 0x10, 2, 1);
    print_pgp_lane("linkRdy", 0x84, 1, 1);
    
    print_pgp_lane("rxFrameCnt", 0x24, 0, 0xffffffff);
    print_pgp_lane("rxFrameErrCnt", 0x28, 0, 0xf);

    printf("\n");
    print_pgp_lane("txFrameCnt" , 0x90, 0, 0xffffffff);
}

void AxisG2Device::setup_lanes(int lane_mask)
{
    for (int i=0; i<MAX_LANES; i++) {
        if (lane_mask & (1<<i)) {
            set_reg32(pci_resource, DMA_LANES(i) + CLIENT, 0);
            set_reg32(pci_resource, PGP_LANES(i) + TX_CONTROL, 1);
        }
    }   
}

void AxisG2Device::loop_test(int lane_mask, int loopb, int size, int op_code, int fifolo)
{
    if (loopb >= 0) {
        for(int i=0; i<4; i++) {
            set_reg32(pci_resource, PGP_LANES(i) + LOOPBACK, (loopb & (1<<i)) ? (2<<16) : 0);
        }
    }   
    int tx_req_delay = 0;
    uint32_t control = get_reg32(pci_resource, PGP_TX_SIM + CONTROL);
    printf("AppTxSim control = %08x\n", control);

    control = ((tx_req_delay & 0x0F) << 24) | ((fifolo & 0x0F) << 28); 
    set_reg32(pci_resource, PGP_TX_SIM + CONTROL, control);
    set_reg32(pci_resource, PGP_TX_SIM + SIZE, size);

    control |= ((lane_mask & 0xff) << 0) | ((op_code > 0x7f ? 0 : ((op_code << 1) | 1)) << 8);
    set_reg32(pci_resource, PGP_TX_SIM + CONTROL, control);

    printf("AppTxSim control = %08x\n", control);

}

DmaBuffer* AxisG2Device::read()
{
    volatile uint64_t dma_data;
    // poll for new event
    do {
        dma_data = ((volatile uint64_t*)rx_desc)[read_index];
    }
    while (dma_data == 0);

    uint32_t buffer_index = (dma_data >> 4) & 0xFFFFF;
    DmaBuffer* buffer = fifo[buffer_index];

    buffer->size  = (dma_data >> 32) & 0xFFFFFF;
    uint32_t dest = (dma_data >> 56) & 0xFF;
    buffer->dest = dest >> 5;
    
    // refill buffer with new buffer from big memory pool and return to hardware
    pool->buffer_queue.pop(fifo[buffer_index]);
    ((uint64_t*)rx_desc)[read_index] = 0;
    set_reg32(pci_resource, CLIENTS(0) + DESC_FIFO_LO, fifo[buffer_index]->phys & 0xFFFFFFFF);
    set_reg32(pci_resource, CLIENTS(0) + DESC_FIFO_HI, (buffer_index << 8) | ((fifo[buffer_index]->phys >> 32) & 0x000000FF)); 
    read_index = (read_index+1) & (RX_DESC_SIZE - 1);
    
    return buffer;
}

