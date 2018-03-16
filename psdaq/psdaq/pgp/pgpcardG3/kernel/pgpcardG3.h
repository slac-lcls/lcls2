//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card G3
// Project       : 8 lane PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : pgpcardG3.h
// Author        : jackp@slac.stanford.edu
// Created       : 2015.3.24
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2015 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/cdev.h>
#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/spinlock.h>
#include <asm/spinlock.h>

#define RHEL7 1

#define NUMBER_OF_LANES 8
#define NUMBER_OF_VC 4
#define MAXIMUM_NUMBER_LANES_PER_CLIENT NUMBER_OF_LANES

#include "../include/PgpCardMod.h"
#include "../include/PgpCardG3Status.h"

// DMA Buffer Size, Bytes
#define DEF_RX_BUF_SIZE 2097152
#define DEF_TX_BUF_SIZE 2097152

// Number of RX & TX Buffers
#define NUMBER_OF_RX_BUFFERS 512
//#define NUMBER_OF_RX_BUFFERS 32
//#define MINIMUM_FIRMWARE_BUFFER_COUNT_THRESHOLD 4
#define NUMBER_OF_RX_CLIENT_BUFFERS ((NUMBER_OF_RX_BUFFERS*MAXIMUM_NUMBER_LANES_PER_CLIENT)/NUMBER_OF_LANES)
#define NUMBER_OF_TX_BUFFERS 128
#define DEF_TX_QUEUE_CNT (NUMBER_OF_TX_BUFFERS)

#define ALMOST_FULL_THRESH 8
#define ALMOST_FULL_ASSERT 0x1FF

// PCI IDs
#define PCI_VENDOR_ID_XILINX      0x10ee
#define PCI_DEVICE_ID_XILINX_PCIE 0x0007
#define PCI_VENDOR_ID_SLAC           0x1a4a
#define PCI_DEVICE_ID_SLAC_PGPCARD   0x2020

// Max number of devices to support
#define MAX_PCI_DEVICES 8

// Module Name
#define MOD_NAME "pgpcardG3"
#define PGPCARD_VERSION "pgpcardG3 driver v02.02.13"

#define IRQ_HISTO_SIZE 1024

//   for multiple port devices we now have an add next port IOCTL command.
#define NUMBER_OF_MINOR_DEVICES (NUMBER_OF_LANES + 1) 
#define ALL_LANES_MASK ((1<<NUMBER_OF_LANES)-1)
#define NUMBER_OF_LANE_CLIENTS (NUMBER_OF_LANES*2)
#define MAX_NUMBER_OPEN_CLIENTS (NUMBER_OF_LANE_CLIENTS + 1) // One for the back door
#define NOT_INTIALIZED 0xffffff
enum MODELS {SmallMemoryModel=4, LargeMemoryModel=8};

// Address Map, offset from base
struct PgpCardG3Reg {
    __u32 version;       // Software_Addr = 0x000,        Firmware_Addr(13 downto 2) = 0x000
    __u32 serNumLower;   // Software_Addr = 0x004,        Firmware_Addr(13 downto 2) = 0x001
    __u32 serNumUpper;   // Software_Addr = 0x008,        Firmware_Addr(13 downto 2) = 0x002
    __u32 scratch;       // Software_Addr = 0x00C,        Firmware_Addr(13 downto 2) = 0x003
    __u32 cardRstStat;   // Software_Addr = 0x010,        Firmware_Addr(13 downto 2) = 0x004
    __u32 irq;           // Software_Addr = 0x014,        Firmware_Addr(13 downto 2) = 0x005
    __u32 pgpRate;       // Software_Addr = 0x018,        Firmware_Addr(13 downto 2) = 0x006
    __u32 sysSpare0;     // Software_Addr = 0x01C,        Firmware_Addr(13 downto 2) = 0x007
    __u32 txOpCode;      // Software_Addr = 0x020,        Firmware_Addr(13 downto 2) = 0x008
    __u32 txLocPause;    // Software_Addr = 0x024,        Firmware_Addr(13 downto 2) = 0x009
    __u32 txLocOvrFlow;  // Software_Addr = 0x028,        Firmware_Addr(13 downto 2) = 0x00A
    __u32 pciStat[4];    // Software_Addr = 0x038:0x02C,  Firmware_Addr(13 downto 2) = 0x00E:0x00B
    __u32 sysSpare1;     // Software_Addr = 0x03C,        Firmware_Addr(13 downto 2) = 0x00F

    __u32 evrCardStat[3];// Software_Addr = 0x048:0x040,  Firmware_Addr(13 downto 2) = 0x012:0x010
    __u32 evrLinkErrorCount; // Software_Addr = 0x04C,    Firmware_Addr ????
    __u32 evrFiducial;   // Software_addr = 0x050,
    __u32 evrSpare0[11]; // Software_Addr = 0x07C:0x054,  Firmware_Addr(13 downto 2) = 0x01F:0x013

   __u32 pgpCardStat[2];// Software_Addr = 0x084:0x080,  Firmware_Addr(13 downto 2) = 0x021:0x020
    __u32 pgpSpare0[54]; // Software_Addr = 0x15C:0x088,  Firmware_Addr(13 downto 2) = 0x057:0x022

    __u32 fiducials[NUMBER_OF_LANES]; // Software_Addr = 0x17C:0x160,  Firmware_Addr(13 downto 2) = 0x05F:0x058
    __u32 runCode[NUMBER_OF_LANES];   // Software_Addr = 0x19C:0x180,  Firmware_Addr(13 downto 2) = 0x067:0x060
    __u32 acceptCode[NUMBER_OF_LANES];// Software_Addr = 0x1BC:0x1A0,  Firmware_Addr(13 downto 2) = 0x06F:0x068

    __u32 runDelay[NUMBER_OF_LANES];   // Software_Addr = 0x1DC:0x1C0,  Firmware_Addr(13 downto 2) = 0x077:0x070
    __u32 acceptDelay[NUMBER_OF_LANES];// Software_Addr = 0x1FC:0x1E0,  Firmware_Addr(13 downto 2) = 0x07F:0x078

    __u32 pgpLaneStat[NUMBER_OF_LANES];// Software_Addr = 0x21C:0x200,  Firmware_Addr(13 downto 2) = 0x087:0x080
    __u32 evrRunCodeCount[NUMBER_OF_LANES]; // Software_Addr = 0x23C:0x220, Firmware_Addr ????
    __u32 LutDropCnt[NUMBER_OF_LANES]; // Software_addr = ox25C:0x240, Firmware_addr ????
    __u32 AcceptCnt[NUMBER_OF_LANES]; // Software addr = 0x27C:0x260, Firmware_addr ????
    __u32 pgpSpare1[32]; // Software_Addr = 0x2FC:0x280,  Firmware_Addr(13 downto 2) = 0x0BF:0x088
    __u32 BuildStamp[64];// Software_Addr = 0x3FC:0x300,  Firmware_Addr(13 downto 2) = 0x0FF:0x0C0

    //PciRxDesc.vhd
    __u32 rxFree[NUMBER_OF_LANES];     // Software_Addr = 0x41C:0x400,  Firmware_Addr(13 downto 2) = 0x107:0x100
    __u32 rxSpare0[24];  // Software_Addr = 0x47C:0x420,  Firmware_Addr(13 downto 2) = 0x11F:0x108
    __u32 rxFreeStat[NUMBER_OF_LANES]; // Software_Addr = 0x49C:0x480,  Firmware_Addr(13 downto 2) = 0x127:0x120
    __u32 rxSpare1[24];  // Software_Addr = 0x4FC:0x4A0,  Firmware_Addr(13 downto 2) = 0x13F:0x128
    __u32 rxMaxFrame;    // Software_Addr = 0x500,        Firmware_Addr(13 downto 2) = 0x140
    __u32 rxCount;       // Software_Addr = 0x504,        Firmware_Addr(13 downto 2) = 0x141
    __u32 rxStatus;      // Software_Addr = 0x508,        Firmware_Addr(13 downto 2) = 0x142
    __u32 rxRead[2];     // Software_Addr = 0x510:0x50C,  Firmware_Addr(13 downto 2) = 0x144:0x143
    __u32 rxSpare2[187]; // Software_Addr = 0x77C:0x514,  Firmware_Addr(13 downto 2) = 0x1FF:0x145

    //PciTxDesc.vhd
    __u32 txWrA[8];      // Software_Addr = 0x81C:0x800,  Firmware_Addr(13 downto 2) = 0x207:0x200
    __u32 txSpare0[24];  // Software_Addr = 0x87C:0x820,  Firmware_Addr(13 downto 2) = 0x21F:0x208
    __u32 txWrB[8];      // Software_Addr = 0x89C:0x880,  Firmware_Addr(13 downto 2) = 0x227:0x220
    __u32 txSpare1[24];  // Software_Addr = 0x8FC:0x8A0,  Firmware_Addr(13 downto 2) = 0x23F:0x228
    __u32 txCount[8];    // Software_Addr = 0x900,        Firmware_Addr(13 downto 2) = 0x240
    __u32 txAFull;       // Software_Addr = 0x920,        Firmware_Addr(13 downto 2) = 0x248
    __u32 txControl;     // Software_Addr = 0x924,        Firmware_Addr(13 downto 2) = 0x249
                         // txClear[23:16], txEnable[7:0]
    /* __u32 txStat[2];     // Software_Addr = 0x904:0x900,  Firmware_Addr(13 downto 2) = 0x241:0x240 */
    /* __u32 txCount;       // Software_Addr = 0x908,        Firmware_Addr(13 downto 2) = 0x242 */
    /* __u32 txRead;        // Software_Addr = 0x90C,        Firmware_Addr(13 downto 2) = 0x243 */
};

// Structure for TX buffers
struct TxBuffer {
   dma_addr_t  dma;
   unchar*     buffer;
   __u32       lane;
   __u32       vc;
   __u32       length;
   __u32       allocated;
};

// Structure for RX buffers
struct RxBuffer {
   dma_addr_t  dma;
   unchar*     buffer;
   __u32       lengthError;
   __u32       fifoError;
   __u32       eofe;
   __u32       lane;
   __u32       vc;
   __u32       length;
   __u32       index;
};

// Client structure
struct Client {
    __u32             mask;
    __u32             vcMask;
    struct file*      fp;
    struct inode*     inode;
    __u32             shared;
    // Queues
    wait_queue_head_t inq;
    wait_queue_head_t outq;
};

// Device structure
struct PgpDevice {

   // PCI address regions
   ulong             baseHdwr;
   ulong             baseLen;
   struct PgpCardG3Reg *reg;

   // Device structure
   int          major;
   struct Client client[MAX_NUMBER_OPEN_CLIENTS];
   struct cdev  cdev;

   // Async queue
   struct fasync_struct *async_queue;

   __u32 isOpen;    // lanes open mask
   __u32 openCount;

   // Debug flag
   __u32 debug;

  struct tasklet_struct dma_task;

   // IRQ
   int irq;

   // Top pointer for rx queue
   struct RxBuffer** rxBuffer;
   struct RxBuffer** rxQueue[NUMBER_OF_LANE_CLIENTS];
   __u32            rxRead[NUMBER_OF_LANE_CLIENTS];
   __u32            rxWrite[NUMBER_OF_LANE_CLIENTS];
   __u32            rxBufferCount;
   __u32*           rxHisto;
   __u32*           rxLoopHisto;
   __u32            rxLaneHisto[NUMBER_OF_LANE_CLIENTS][NUMBER_OF_VC];
   __u32*           rxBuffersHisto;
   __u32            rxCopyToUserPrintCount;
   __u32            rxTossedBuffers[NUMBER_OF_LANE_CLIENTS];
   __u32            rxTotalTossedBuffers;
   __u32            irqRetryAccumulator;
   __u32*           irqRetryHisto;

   // Top pointer for tx queue
   __u32             txBufferCount;
   struct TxBuffer** txBuffer;
   __u32             txRead;
   spinlock_t*       txLock;
   spinlock_t*       txLockIrq;
   spinlock_t*       rxLock;
   spinlock_t*       readLock;
   spinlock_t*       ioctlLock;
   spinlock_t*       releaseLock;
   spinlock_t*       pollLock;
   __u32             goingDown;
   __u32             pollEnabled;
   __u32*            txHisto;
   __u32*            txHistoLV;
   __u32*            txLocPauseHisto;
   __u32*            txLocOvrFlowHisto;
   __u32             interruptNesting;
   __u32             noClientPacketCount[NUMBER_OF_LANES];
   __u32             noClientPacketMax;
   __u32             cfrbmesgCount;
   __u32             eofeMesgCount;
   __u32             laneIndexes[NUMBER_OF_LANES];
   __u32             orderMessageCount;
   PgpCardG3Status*  status;
};

// TX32 Structure
typedef struct {
    // Data
    __u32 model; // large=8, small=4
    __u32 cmd; // ioctl commands
    __u32 data;

    // Lane & VC
   __u32 pgpLane;
   __u32 pgpVc;

   __u32   size;  // dwords

} PgpCardG3Tx32;

// RX32 Structure
typedef struct {
    __u32   model; // large=8, small=4
    __u32   maxSize; // dwords
    __u32   data;

   // Lane & VC
   __u32    pgpLane;
   __u32    pgpVc;

   // Data
   __u32    rxSize;  // dwords

   // Error flags
   __u32   eofe;
   __u32   fifoErr;
   __u32   lengthErr;

} PgpCardG3Rx32;

// Function prototypes
int PgpCardG3_Open(struct inode *inode, struct file *filp);
int PgpCardG3_Release(struct inode *inode, struct file *filp);
ssize_t PgpCardG3_Write(struct file *filp, const char *buf, size_t count, loff_t *f_pos);
ssize_t PgpCardG3_Read(struct file *filp, char *buf, size_t count, loff_t *f_pos);
int PgpCardG3_Ioctl(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg);
int my_Ioctl(struct file *filp, __u32 cmd, __u64 argument);
unsigned countRXFirmwareBuffers(struct PgpDevice*, __u32);
unsigned countRxBuffers(struct PgpDevice*, __u32);
unsigned countTxBuffers(struct PgpDevice*);
static irqreturn_t PgpCardG3_IRQHandler(int irq, void *dev_id, struct pt_regs *regs);
static unsigned int PgpCardG3_Poll(struct file *filp, poll_table *wait );
static int PgpCardG3_Probe(struct pci_dev *pcidev, const struct pci_device_id *dev_id);
static void PgpCardG3_Remove(struct pci_dev *pcidev);
static int PgpCardG3_Init(void);
static void PgpCardG3_Exit(void);
int dumpWarning(struct PgpDevice *);
int PgpCardG3_Mmap(struct file *filp, struct vm_area_struct *vma);
int PgpCardG3_Fasync(int fd, struct file *filp, int mode);
void PgpCardG3_VmOpen(struct vm_area_struct *vma);
void PgpCardG3_VmClose(struct vm_area_struct *vma);

// PCI device IDs
static struct pci_device_id PgpCardG3_Ids[] = {
   { PCI_DEVICE(PCI_VENDOR_ID_XILINX,PCI_DEVICE_ID_XILINX_PCIE) },
   { PCI_DEVICE(PCI_VENDOR_ID_SLAC,   PCI_DEVICE_ID_SLAC_PGPCARD)   },
   { 0, }
};

// PCI driver structure
static struct pci_driver PgpCardG3Driver = {
  .name     = MOD_NAME,
  .id_table = PgpCardG3_Ids,
  .probe    = PgpCardG3_Probe,
  .remove   = PgpCardG3_Remove,
};

// Define interface routines
struct file_operations PgpCardG3_Intf = {
   read:    PgpCardG3_Read,
   write:   PgpCardG3_Write,
#ifndef RHEL7
   ioctl:   PgpCardG3_Ioctl,
#endif
   open:    PgpCardG3_Open,
   release: PgpCardG3_Release,
   poll:    PgpCardG3_Poll,
   fasync:  PgpCardG3_Fasync,
   mmap:    PgpCardG3_Mmap,
};

// Virtual memory operations
static struct vm_operations_struct PgpCardG3_VmOps = {
  open:  PgpCardG3_VmOpen,
  close: PgpCardG3_VmClose,
};

