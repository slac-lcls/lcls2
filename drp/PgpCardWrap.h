//---------------------------------------------------------------------------------
// Title         : Kernel Module For PGP To PCI Bridge Card
// Project       : PGP To PCI-E Bridge Card
//---------------------------------------------------------------------------------
// File          : PgpCardWrap.h
// Author        : Ryan Herbst, rherbst@slac.stanford.edu
// Created       : 05/18/2010
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Copyright (c) 2010 by SLAC National Accelerator Laboratory. All rights reserved.
//---------------------------------------------------------------------------------
// Modification history:
// 05/18/2010: created.
//---------------------------------------------------------------------------------
#ifndef __PGP_CARD_WRAP_H__
#define __PGP_CARD_WRAP_H__

#include <linux/types.h>
#include "PgpCardMod.h"

// Send Frame, size in dwords
// int pgpcard_send(int fd, void *buf, size_t count, uint lane, uint vc);

// Send Frame, size in dwords, return in dwords
// int pgpcard_recv(int fd, void *buf, size_t maxSize, uint *lane, uint *vc, uint *eofe, uint *fifoErr, uint *lengthErr);

// Read Status
// int pgpcard_status(int fd, PgpCardStatus *status);

// Set debug
// int pgpcard_setDebug(int fd, uint level);

// Set/Clear RX Reset For Lane
// int pgpcard_setRxReset(int fd, uint lane);
// int pgpcard_clrRxReset(int fd, uint lane);

// Set/Clear TX Reset For Lane
// int pgpcard_setTxReset(int fd, uint lane);
// int pgpcard_clrTxReset(int fd, uint lane);

// Set/Clear Loopback For Lane
// int pgpcard_setLoop(int fd, uint lane);
// int pgpcard_clrLoop(int fd, uint lane);

// Reset Counters
// int pgpcard_rstCount(int fd);

// Dump Debug
// int pgpcard_dumpDebug(int fd);


// Send Frame, size in dwords
inline int pgpcard_send(int fd, void *buf, size_t size, uint lane, uint vc) {
   PgpCardTx pgpCardTx;

   pgpCardTx.model   = (sizeof(buf));
   pgpCardTx.cmd     = IOCTL_Normal_Write;
   pgpCardTx.pgpVc   = vc;
   pgpCardTx.pgpLane = lane;
   pgpCardTx.size    = size;
   pgpCardTx.data    = (__u32*)buf;

   return(write(fd,&pgpCardTx,sizeof(PgpCardTx)));
}


// Send Frame, size in dwords, return in dwords
inline int pgpcard_recv(int fd, void *buf, size_t maxSize, uint *lane, uint *vc, uint *eofe, uint *fifoErr, uint *lengthErr) {
   PgpCardRx pgpCardRx;
   int       ret;

   pgpCardRx.maxSize = maxSize;
   pgpCardRx.data    = (__u32*)buf;
   pgpCardRx.model   = sizeof(buf);

   ret = read(fd,&pgpCardRx,sizeof(PgpCardRx));

   *lane      = pgpCardRx.pgpLane;
   *vc        = pgpCardRx.pgpVc;
   *eofe      = pgpCardRx.eofe;
   *fifoErr   = pgpCardRx.fifoErr;
   *lengthErr = pgpCardRx.lengthErr;

   return(ret);
}


// Read Status
inline int pgpcard_status(int fd, PgpCardStatus *status) {

   // the buffer is a PgpCardTx on the way in and a PgpCardStatus on the way out
   __u8*      c = (__u8*) status;  // this adheres to strict aliasing rules
   PgpCardTx* p = (PgpCardTx*) c;

   p->model = sizeof(p);
   p->cmd   = IOCTL_Read_Status;
   p->data  = (__u32*)status;
   return(write(fd, p, sizeof(PgpCardStatus)));
}


// Set debug
inline int pgpcard_setDebug(int fd, uint level) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Set_Debug;
   t.data  = (__u32*) level;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

// Set/Clear RX Reset For Lane
inline int pgpcard_setRxReset(int fd, uint lane) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Set_Rx_Reset;;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

inline int pgpcard_clrRxReset(int fd, uint lane){
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Clr_Rx_Reset;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

// Set/Clear TX Reset For Lane
inline int pgpcard_setTxReset(int fd, uint lane) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Set_Tx_Reset;;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));

}

inline int pgpcard_clrTxReset(int fd, uint lane) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Clr_Tx_Reset;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

// Set/Clear Loopback For Lane
inline int pgpcard_setLoop(int fd, uint lane) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Set_Loop;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

inline int pgpcard_clrLoop(int fd, uint lane) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Clr_Loop;
   t.data  = (__u32*) lane;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

// Reset Counters
inline int pgpcard_rstCount(int fd) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Count_Reset;
   t.data  = (__u32*)0;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

// Dump Debug
inline int pgpcard_dumpDebug(int fd) {
   PgpCardTx  t;

   t.model = sizeof(PgpCardTx*);
   t.cmd   = IOCTL_Dump_Debug;
   return(write(fd, &t, sizeof(PgpCardTx)));
}

#endif
