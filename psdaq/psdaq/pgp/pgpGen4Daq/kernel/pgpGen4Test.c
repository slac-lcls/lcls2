//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card
// Project       : PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : pgpcardG3.c
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
#include <linux/compat.h>
#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/types.h>

#include "pgpGen4Daq.h"
#include "../include/DmaDriver.h"

// Module Name
#define MOD_NAME "pgpGen4Test"
#define PGPCARD_VERSION "pgpGen4Test driver v00.00.00"

// PCI device IDs
static struct pci_device_id PgpGen4Test_Ids[] = {
   { PCI_DEVICE(PCI_VENDOR_ID_SLAC,   PCI_DEVICE_ID_SLAC_PRI)   },
   { PCI_DEVICE(PCI_VENDOR_ID_SLAC,   PCI_DEVICE_ID_SLAC_SEC)   },
   { 0, }
};

// Function prototypes
int     PgpGen4Test_Open(struct inode *inode, struct file *filp);
int     PgpGen4Test_Release(struct inode *inode, struct file *filp);
ssize_t PgpGen4Test_Write(struct file *filp, const char *buf, size_t count, loff_t *f_pos);
ssize_t PgpGen4Test_Read(struct file *filp, char *buf, size_t count, loff_t *f_pos);
int     PgpGen4Test_Ioctl(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg);
static irqreturn_t PgpGen4Test_IRQHandler(int irq, void *dev_id, struct pt_regs *regs);
static unsigned int PgpGen4Test_Poll(struct file *filp, poll_table *wait );
static int PgpGen4Test_Probe(struct pci_dev *pcidev, const struct pci_device_id *dev_id);
static void PgpGen4Test_Remove(struct pci_dev *pcidev);
static int  PgpGen4Test_Init(void);
static void PgpGen4Test_Exit(void);
int  PgpGen4Test_Mmap(struct file *filp, struct vm_area_struct *vma);
int  PgpGen4Test_Fasync(int fd, struct file *filp, int mode);
void PgpGen4Test_VmOpen(struct vm_area_struct *vma);
void PgpGen4Test_VmClose(struct vm_area_struct *vma);


// PCI driver structure
static struct pci_driver PgpGen4TestDriver = {
  .name     = MOD_NAME,
  .id_table = PgpGen4Test_Ids,
  .probe    = PgpGen4Test_Probe,
  .remove   = PgpGen4Test_Remove,
};

// Define interface routines
struct file_operations PgpGen4Test_Intf = {
   read:    PgpGen4Test_Read,
   write:   PgpGen4Test_Write,
#ifndef RHEL7
   ioctl:   PgpGen4Test_Ioctl,
#endif
   open:    PgpGen4Test_Open,
   release: PgpGen4Test_Release,
   poll:    PgpGen4Test_Poll,
   fasync:  PgpGen4Test_Fasync,
   mmap:    PgpGen4Test_Mmap,
};

// Virtual memory operations
static struct vm_operations_struct PgpGen4Test_VmOps = {
  open:  PgpGen4Test_VmOpen,
  close: PgpGen4Test_VmClose,
};

MODULE_LICENSE("GPL");
MODULE_DEVICE_TABLE(pci, PgpGen4Test_Ids);
module_init(PgpGen4Test_Init);
module_exit(PgpGen4Test_Exit);

#define SUCCESS 0
#define ERROR -1

#define MON_BUFFER_SIZE 0x10000
#define RX_BUFFER_SIZE (2*1024*1024)
#define RX_BUFFERS 0x80
#define RX_DESC_SIZE 0x1000

// Global Variable
struct DaqDevice gDaqDevices[MAX_PCI_DEVICES];


// Open Returns 0 on success, error code on failure
int PgpGen4Test_Open(struct inode *inode, struct file *filp) {
  struct DaqDevice *dev;
  int ret = SUCCESS;

  // Extract structure for card
  dev = container_of(inode->i_cdev, struct DaqDevice, cdev);
  filp->private_data = &dev->client[iminor(inode)];

  printk(KERN_DEBUG"%s: Maj %u Open major %u minor %u\n", MOD_NAME, dev->major, imajor(inode), iminor(inode));

  dev->openCount += 1;

  return ret;
}


// PgpGen4Test_Release
// Called when the device is closed
// Returns 0 on success, error code on failure
int PgpGen4Test_Release(struct inode *inode, struct file *filp) {
  struct Client*client = (struct Client *)filp->private_data;
  struct DaqDevice *dev = (struct DaqDevice*)(client->dev);

  dev->openCount -= 1;

  return SUCCESS;
}


// PgpGen4Test_Write
// Called when the device is written to
// Returns write count on success. Error code on failure.
ssize_t PgpGen4Test_Write(struct file *filp, const char* buffer, size_t count, loff_t* f_pos) {
  int i,j;
  struct Client*client = (struct Client *)filp->private_data;
  struct DaqDevice *dev = (struct DaqDevice*)(client->dev);

  printk(KERN_WARNING "%s: Write:  Releasing memory. Maj=%i\n",
         MOD_NAME, dev->major);

  dma_free_coherent(dev->device, MON_BUFFER_SIZE, dev->monAddr, dev->monHandle);

  for(j=0; j<NUMBER_OF_CLIENTS; j++) {
    client = &dev->client[j];

    for(i=0; i<RX_BUFFERS; i++)
      dma_free_coherent(dev->device, RX_BUFFER_SIZE, 
                        client->rxBuffer[i], client->rxHandle[i]);

    dma_free_coherent(dev->device, 8*RX_DESC_SIZE, 
                      client->rxDescPage, client->rxDescHandle);
    kfree(client->rxBuffer);
    kfree(client->rxHandle);
  }

  printk(KERN_WARNING "%s: Write:  User reset. Maj=%i\n",
         MOD_NAME, dev->major);

  // Reset the pipelines (auto clears)
  dev->reg->reset = 1;

  // Write scratchpad
  dev->reg->scratch = SPAD_WRITE;

  dev->monAddr = dma_alloc_coherent(dev->device, 
                                    MON_BUFFER_SIZE,
                                    &(dev->monHandle), 
                                    GFP_KERNEL);
  printk(KERN_INFO"%s: Write: monAddr %p mapped to %p\n", MOD_NAME, dev->monAddr, (void*)dev->monHandle);

  iowrite32((dev->monHandle>> 0)&0xffffffff, (__u32*)dev->reg+(0x00800014>>2));
  iowrite32((dev->monHandle>>32)&0x000000ff, (__u32*)dev->reg+(0x00800018>>2));

  for(j=0; j<NUMBER_OF_CLIENTS; j++) {
    client = &dev->client[j];
    client->reg = (struct ClientReg*)(&dev->reg->clients[j]);

    // Allocate receive descriptor page (for each application)
    // DMA completion writes to this page
    client->rxDescPage   = 
      dma_alloc_coherent(dev->device,
                         8*RX_DESC_SIZE,
                         &(client->rxDescHandle),
                         GFP_KERNEL);

    printk(KERN_INFO"%s: Write: descPage %p mapped to %p\n", MOD_NAME, client->rxDescPage, (void*)client->rxDescHandle);

    memset(client->rxDescPage, 0, 8*RX_DESC_SIZE);
    iowrite32((client->rxDescHandle>> 0)&0xffffffff, 
              (__u32*)&client->reg->descAddrLo);
    iowrite32((client->rxDescHandle>>32)&0x000000ff, 
              (__u32*)&client->reg->descAddrHi);

    // Allocate receive buffers
    client->rxBuffer = kmalloc(RX_BUFFERS*sizeof(void*)     ,GFP_KERNEL);
    client->rxHandle = kmalloc(RX_BUFFERS*sizeof(dma_addr_t),GFP_KERNEL);
    for(i=0; i<RX_BUFFERS; i++) {
      client->rxBuffer[i] = dma_alloc_coherent(dev->device,
                                               RX_BUFFER_SIZE,
                                               &(client->rxHandle[i]),
                                               GFP_KERNEL);
      if (client->rxBuffer[i]==NULL) {
        printk(KERN_WARNING"%s: Write: dma_alloc_coherent failed at index %u. Maj=%i.\n", MOD_NAME, i, dev->major);
        break;
      }
      
      if (i == 0) {
        printk(KERN_WARNING"%s: Write: client %i first buffer at %p. Maj=%i.\n", MOD_NAME, j, client->rxBuffer[i], dev->major);
      }

      //  Write bus address and index
      iowrite32((client->rxHandle[i]>> 0)&0xffffffff, 
                (__u32*)&client->reg->descFifoLo);
      iowrite32((i<<8) | ((client->rxHandle[i]>>32)&0x000000ff),
                (__u32*)&client->reg->descFifoHi);
    }

    client->readIndex = 0;
    client->dev       = dev;
  }

  return 0;
}


// PgpGen4Test_Read
// Called when the device is read from
// Returns read count on success. Error code on failure.
ssize_t PgpGen4Test_Read(struct file *filp, char *buffer, size_t count, loff_t *f_pos) {
  ssize_t            ret;
  ssize_t            res;
  struct Client*  client = (struct Client *)filp->private_data;
  struct DaqDevice*  dev = client->dev;
 
  struct DmaReadData rd;

  // Verify that size of passed structure
  if ( count != sizeof(struct DmaReadData) ) {
    dev_warn(dev->device,
             "Read: Called with incorrect size. Got=%li, Exp=%li\n",
             count,sizeof(struct DmaReadData));
    return(-1);
  }

  // Copy read structure
  if ( (ret=copy_from_user(&rd,buffer,sizeof(struct DmaReadData)))) {
    dev_warn(dev->device,
             "Read: failed to copy struct from user space ret=%li, user=%p kern=%p\n",
             ret, (void *)buffer, (void *)&rd);
    return -1;
  }

  if (rd.index & (1<<31)) {
    //
    //  Read monitoring buffer
    //
    if ( (ret=copy_to_user((void*)rd.data, dev->monAddr, MON_BUFFER_SIZE) )) {
      dev_warn(dev->device,
               "Read: failed to copy data to user space ret=%li,\
 user=%p kern=%p size=%u.\n",
               ret, (void*)rd.data, dev->monAddr, MON_BUFFER_SIZE);
      res = -1;
    }
  }

  else {
    //
    //  Read descriptor
    //
    //    uint64_t desc = ((uint64_t*)dev->rxDescPage)[dev->rdIndex++];
    //    unsigned descIndex = rd.index;
    unsigned descIndex = client->readIndex & (RX_DESC_SIZE-1);
    uint64_t desc = ((uint64_t*)client->rxDescPage)[descIndex];
    rd.dest  = (desc>>56)&0xff;
    rd.flags = (desc>>24)&0xff;
    rd.index = (desc>> 4)&0xfffff;
    rd.error = (desc>> 0)&0x7;
    rd.size  = (desc>>32)&0xffffff;

    if (rd.size == 0) {
      //      dev_warn(dev->device,"Read: descriptor %u is not ready\n", descIndex);
      //      res = -1
      //
      //  For testing, descriptor page is failing.  Continue without it.
      //
    }

    //
    //  Read data buffer
    //
    else if (rd.index >= RX_BUFFERS) {
      dev_warn(dev->device,
               "Read: descriptor gives buffer index out of range (%u) [%llx].\n",
               rd.index, desc);
      res = -1;
    }
    else {
      if ( (ret=copy_to_user((void*)rd.data, client->rxBuffer[rd.index], rd.size) )) {
        dev_warn(dev->device,
                 "Read: failed to copy data to user space ret=%li,\
 user=%p kern=%p size=%u.\n",
               ret, (void*)rd.data, client->rxBuffer[rd.index], rd.size);
        res = -1;
      }
      //  Clear the descriptor
      ((uint64_t*)client->rxDescPage)[descIndex] = 0;
      //  Queue buffer index to descriptor FIFO
      iowrite32((client->rxHandle[rd.index]>> 0)&0xffffffff, 
                (__u32*)&client->reg->descFifoLo);
      iowrite32((rd.index<<8) | ((client->rxHandle[rd.index]>>32)&0x000000ff),
                (__u32*)&client->reg->descFifoHi);
      client->readIndex++;
    }
  }

  if ( (ret=copy_to_user(buffer, &rd, sizeof(rd))) ) {
    dev_warn(dev->device,
             "Read: failed to copy struct to user space ret=%li, user=%p, kern=%p\n",
             ret, (void*)buffer, (void*)&rd);
    res = -1;
  }
  else
    res = ret;

  return res;
}

// IRQ Handler
static irqreturn_t PgpGen4Test_IRQHandler(int irq, void *dev_id, struct pt_regs *regs) {
  return(IRQ_NONE);
}

// Poll/Select
static __u32 PgpGen4Test_Poll(struct file *filp, poll_table *wait ) {
  __u32 mask    = 0;
  return(mask);
}


// Probe device
static int PgpGen4Test_Probe(struct pci_dev *pcidev, const struct pci_device_id *dev_id) {
  int i, j, res, ret;
  dev_t chrdev = 0;
  struct DaqDevice *dev;
  struct Client *client;
  struct pci_device_id *id = (struct pci_device_id *) dev_id;

  // We keep device instance number in id->driver_data
  id->driver_data = -1;

  // Find empty structure
  for (i = 0; i < MAX_PCI_DEVICES; i++) {
    if (gDaqDevices[i].baseHdwr == 0) {
      id->driver_data = i;
      break;
    }
  }

  // Overflow
  if (id->driver_data < 0) {
    printk(KERN_WARNING "%s: Probe: Too Many Devices.\n", MOD_NAME);
    return -EMFILE;
  }
  dev = &gDaqDevices[id->driver_data];

  // Allocate device numbers for character device.
  res = alloc_chrdev_region(&chrdev, 0, NUMBER_OF_MINOR_DEVICES, MOD_NAME);
  if (res < 0) {
    printk(KERN_WARNING "%s: Probe: Cannot register char device\n", MOD_NAME);
    return res;
  }

  // Init device
  cdev_init(&dev->cdev, &PgpGen4Test_Intf);

  // Initialize device structure
  dev->major = MAJOR(chrdev);

  dev->isOpen        = 0;
  dev->openCount     = 0;
  dev->cdev.owner    = THIS_MODULE;
  dev->cdev.ops      = &PgpGen4Test_Intf;

  // Add device
  if ( cdev_add(&dev->cdev, chrdev, NUMBER_OF_MINOR_DEVICES) ) {
    printk(KERN_WARNING "%s: Probe: Error cdev_adding device Maj=%i with %u devices\n", MOD_NAME,dev->major, NUMBER_OF_MINOR_DEVICES);
  }
  // Enable devices
  ret = pci_enable_device(pcidev);
  if (ret) {
    printk(KERN_WARNING "%s: pci_enable_device() returned %d, Maj %i\n", MOD_NAME, ret, dev->major);
  }

  // Get Base Address of registers from pci structure.
  dev->baseHdwr = pci_resource_start (pcidev, 0);
  dev->baseLen  = pci_resource_len (pcidev, 0);
  printk(KERN_INFO "%s: Probe: Pci Address, Length: %lu, %lu\n", MOD_NAME, dev->baseHdwr, dev->baseLen);

  // Remap the I/O register block so that it can be safely accessed.
  dev->reg = (struct DaqReg *)ioremap_nocache(dev->baseHdwr, dev->baseLen);
  if (! dev->reg ) {
    printk(KERN_WARNING"%s: Probe: Could not remap memory Maj=%i.\n", MOD_NAME,dev->major);
    return (ERROR);
  }

  // Try to gain exclusive control of memory
  if (check_mem_region(dev->baseHdwr, dev->baseLen) < 0 ) {
    printk(KERN_WARNING"%s: Probe: Memory in use Maj=%i.\n", MOD_NAME,dev->major);
    return (ERROR);
  }

  request_mem_region(dev->baseHdwr, dev->baseLen, MOD_NAME);
  printk(KERN_INFO "%s: Probe: Found card. Version=0x%x, Maj=%i\n", MOD_NAME,dev->reg->version,dev->major);

  // Get IRQ from pci_dev structure.
  dev->irq = pcidev->irq;
  printk(KERN_INFO "%s: Probe: IRQ %d Maj=%i\n", MOD_NAME, dev->irq,dev->major);

  // Reset the pipelines (auto clears)
  dev->reg->reset = 1;

  // Write scratchpad
  dev->reg->scratch = SPAD_WRITE;

  dev->device = &(pcidev->dev);

  //  dma_set_mask_and_coherent(dev->device, 0xffffffffffULL);
  dma_set_mask_and_coherent(dev->device, 0x007fffffffULL);

  dev->monAddr = dma_alloc_coherent(dev->device, 
                                    MON_BUFFER_SIZE,
                                    &(dev->monHandle), 
                                    GFP_KERNEL);
  printk(KERN_INFO"%s: Probe: monAddr %p mapped to %p\n", MOD_NAME, dev->monAddr, (void*)dev->monHandle);

  iowrite32((dev->monHandle>> 0)&0xffffffff, (__u32*)dev->reg+(0x00800014>>2));
  iowrite32((dev->monHandle>>32)&0x000000ff, (__u32*)dev->reg+(0x00800018>>2));

  for(j=0; j<NUMBER_OF_CLIENTS; j++) {
    client = &dev->client[j];
    client->reg = (struct ClientReg*)(&dev->reg->clients[j]);

    // Allocate receive descriptor page (for each application)
    // DMA completion writes to this page
    client->rxDescPage   = 
      dma_alloc_coherent(dev->device,
                         8*RX_DESC_SIZE,
                         &(client->rxDescHandle),
                         GFP_KERNEL);

    printk(KERN_INFO"%s: Probe: descPage %p mapped to %p\n", MOD_NAME, client->rxDescPage, (void*)client->rxDescHandle);

    memset(client->rxDescPage, 0, 8*RX_DESC_SIZE);
    iowrite32((client->rxDescHandle>> 0)&0xffffffff, 
              (__u32*)&client->reg->descAddrLo);
    iowrite32((client->rxDescHandle>>32)&0x000000ff, 
              (__u32*)&client->reg->descAddrHi);

    // Allocate receive buffers
    client->rxBuffer = kmalloc(RX_BUFFERS*sizeof(void*)     ,GFP_KERNEL);
    client->rxHandle = kmalloc(RX_BUFFERS*sizeof(dma_addr_t),GFP_KERNEL);
    for(i=0; i<RX_BUFFERS; i++) {
      client->rxBuffer[i] = dma_alloc_coherent(dev->device,
                                               RX_BUFFER_SIZE,
                                               &(client->rxHandle[i]),
                                               GFP_KERNEL);
      if (client->rxBuffer[i]==NULL) {
        printk(KERN_WARNING"%s: Probe: dma_alloc_coherent failed at index %u. Maj=%i.\n", MOD_NAME, i, dev->major);
        break;
      }
      
      if (i == 0) {
        printk(KERN_WARNING"%s: Probe: client %i first buffer at %p. Maj=%i.\n", MOD_NAME, j, client->rxBuffer[i], dev->major);
      }

      //  Write bus address and index
      iowrite32((client->rxHandle[i]>> 0)&0xffffffff, 
                (__u32*)&client->reg->descFifoLo);
      iowrite32((i<<8) | ((client->rxHandle[i]>>32)&0x000000ff),
                (__u32*)&client->reg->descFifoHi);
    }

    client->readIndex = 0;
    client->dev       = dev;
  }
  dev->client[NUMBER_OF_CLIENTS].dev = dev;

  printk(KERN_INFO"%s: Probe: Driver is loaded. Maj=%i %s\n", MOD_NAME,dev->major, PGPCARD_VERSION);
  return SUCCESS;
}


// Remove
static void PgpGen4Test_Remove(struct pci_dev *pcidev) {
  int  i, j;
  struct DaqDevice *dev = NULL;
  struct Client* client;

  // Look for matching device
  for (i = 0; i < MAX_PCI_DEVICES; i++) {
    if ( gDaqDevices[i].baseHdwr == pci_resource_start(pcidev, 0)) {
      dev = &gDaqDevices[i];
      break;
    }
  }

  // Device not found
  if (dev == NULL) {
    printk(KERN_WARNING "%s: Remove: Device Not Found.\n", MOD_NAME);
  }
  else {
    dma_free_coherent(dev->device, MON_BUFFER_SIZE, dev->monAddr, dev->monHandle);

    for(j=0; j<NUMBER_OF_CLIENTS; j++) {
      client = &dev->client[j];

      for(i=0; i<RX_BUFFERS; i++)
        dma_free_coherent(dev->device, RX_BUFFER_SIZE, 
                          client->rxBuffer[i], client->rxHandle[i]);

      dma_free_coherent(dev->device, 8*RX_DESC_SIZE, 
                        client->rxDescPage, client->rxDescHandle);
      kfree(client->rxBuffer);
      kfree(client->rxHandle);
    }

    // Release memory region
    release_mem_region(dev->baseHdwr, dev->baseLen);

    // Unmap
    iounmap(dev->reg);

    // Unregister Device Driver
    cdev_del(&dev->cdev);
    unregister_chrdev_region(MKDEV(dev->major,0), NUMBER_OF_MINOR_DEVICES);

    // Disable device
    //    pci_disable_device(pcidev);  // Xilinx core does not re-enable Bus Master Enable bit
    dev->baseHdwr = 0;
    printk(KERN_INFO"%s: Remove: %s is unloaded. Maj=%i\n", MOD_NAME, PGPCARD_VERSION, dev->major);
  }
}


// Init Kernel Module
static int PgpGen4Test_Init(void) {
  int ret = 0;
  /* Allocate and clear memory for all devices. */
  memset(gDaqDevices, 0, sizeof(struct DaqDevice)*MAX_PCI_DEVICES);

  // Register driver
  ret = pci_register_driver(&PgpGen4TestDriver);

  printk(KERN_INFO"%s: Init: Register driver returned %d\n", MOD_NAME, ret);

  return(ret);
}


// Exit Kernel Module
static void PgpGen4Test_Exit(void) {
  printk(KERN_INFO"%s: Exit.\n", MOD_NAME);
  pci_unregister_driver(&PgpGen4TestDriver);
}

// PgpGen4Test_Ioctl
// Called when ioctl is called on the device
// Returns success.
int PgpGen4Test_Ioctl(struct inode *inode, struct file *filp, __u32 cmd, unsigned long arg) {
  printk(KERN_WARNING "%s: warning Ioctl is deprecated and no longer supported\n", MOD_NAME);
  return SUCCESS;
}

// Memory map
int PgpGen4Test_Mmap(struct file *filp, struct vm_area_struct *vma) {
  
   struct Client*client = (struct Client *)filp->private_data;
   struct DaqDevice *dev = (struct DaqDevice*)(client->dev);

   unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
   unsigned long physical = ((unsigned long) dev->baseHdwr) + offset;
   unsigned long vsize = vma->vm_end - vma->vm_start;
   int result;

   // Check bounds of memory map
   if (vsize > dev->baseLen) {
      printk(KERN_WARNING"%s: Mmap: mmap vsize %08x, baseLen %08x. Maj=%i\n", MOD_NAME,
         (unsigned int) vsize, (unsigned int) dev->baseLen,dev->major);
      return -EINVAL;
   }

   result = io_remap_pfn_range(vma, vma->vm_start, physical >> PAGE_SHIFT,
            vsize, vma->vm_page_prot);
//   result = io_remap_page_range(vma, vma->vm_start, physical, vsize,
//            vma->vm_page_prot);

   if (result) return -EAGAIN;

   vma->vm_ops = &PgpGen4Test_VmOps;
   PgpGen4Test_VmOpen(vma);
   return 0;
}


void PgpGen4Test_VmOpen(struct vm_area_struct *vma) { }


void PgpGen4Test_VmClose(struct vm_area_struct *vma) { }


// Flush queue
int PgpGen4Test_Fasync(int fd, struct file *filp, int mode) {
   struct Client*client = (struct Client *)filp->private_data;
   struct DaqDevice *dev = (struct DaqDevice*)(client->dev);
   return fasync_helper(fd, filp, mode, &(dev->async_queue));
}


