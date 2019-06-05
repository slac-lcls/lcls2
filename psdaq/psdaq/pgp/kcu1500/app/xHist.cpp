/**
 *-----------------------------------------------------------------------------
 * Title      : DMA read utility
 * ----------------------------------------------------------------------------
 * File       : dmaRate.cpp
 * Author     : Ryan Herbst, rherbst@slac.stanford.edu
 * Created    : 2016-08-08
 * Last update: 2016-08-08
 * ----------------------------------------------------------------------------
 * Description:
 * This program will open up a AXIS DMA port and attempt to read data.
 * ----------------------------------------------------------------------------
 * This file is part of the aes_stream_drivers package. It is subject to 
 * the license terms in the LICENSE.txt file found in the top-level directory 
 * of this distribution and at: 
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
 * No part of the aes_stream_drivers package, including this file, may be 
 * copied, modified, propagated, or distributed except according to the terms 
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/

#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <stdlib.h>
#include <argp.h>
#include "AxisDriver.h"
using namespace std;

static void usage(const char* p) {
  printf("Usage: %p [options]\n",p);
  printf("Options:\n");
  printf("\t-d <device file> (default: /dev/datadev_1)\n");
}

static int fd = -1;

static void sigHandler( int signal ) {
  dmaWriteRegister(fd, 0x00800000, 0);
  ::exit(signal);
}

int main (int argc, char **argv) {
   int32_t       ret;
   int32_t       s;
   unsigned      lane = 4;

   const char* dev = "/dev/datadev_1";
   extern char* optarg;
   char c;
   while((c = getopt(argc,argv,"d:l:"))!=EOF) {
     switch(c) {
     case 'd': dev = optarg; break;
     case 'l': lane = strtoul(optarg,NULL,0); break;
     default: usage(argv[0]); exit(1);
     }
   }

   uint8_t*      mask    = new uint8_t [DMA_MASK_SIZE];
   void **       dmaBuffers;
   uint32_t      dmaCount;
   uint32_t      dmaSize;

   dmaInitMaskBytes(mask);
   dmaAddMaskBytes((uint8_t*)mask,(lane<<8));

   if ( (s = open(dev, O_RDWR)) <= 0 ) {
      printf("Error opening %s\n",dev);
      return(1);
   }

   if ( (dmaBuffers = dmaMapDma(s,&dmaCount,&dmaSize)) == NULL ) {
      perror("Failed to map dma buffers!");
      return(0);
   }

   dmaWriteRegister(fd=s, 0x00800000, 1);

   ::signal( SIGINT, sigHandler );

   dmaSetMaskBytes(s,mask);

   unsigned* last = new unsigned[64];
   unsigned* hist = new unsigned[64];
   unsigned* last_hwf = new unsigned[64];
   unsigned* hist_hwf = new unsigned[64];

   unsigned max_ret_cnt = 70000;
   uint32_t      getCnt = max_ret_cnt;
   uint32_t*     rxFlags = new uint32_t[max_ret_cnt];
   uint32_t*     dest    = new uint32_t[max_ret_cnt];
   uint32_t*     dmaIndex = new uint32_t[max_ret_cnt];
   int32_t*      dmaRet   = new int32_t [max_ret_cnt];

   while(1) {

     // DMA Read
     ret = dmaReadBulkIndex(s,getCnt,dmaRet,dmaIndex,rxFlags,NULL,dest);  // 24 usec

     for (int x=0; x < ret; ++x) {
       if ( dmaRet[x] > 0.0 ) {
         const uint32_t* buf = reinterpret_cast<const uint32_t*>(dmaBuffers[dmaIndex[x]]);

         unsigned m=0;
         for(unsigned i=0; i<32; i++) {
           unsigned v=0;
           for(unsigned j=0; j<4; j++)
             v += buf[i+j*257+1];
           hist[m++]=v;
         }
         for(unsigned i=0; i<224;) {
           unsigned v=0;
           for(unsigned j=0; j<8; j++,i++)
             for(unsigned k=0; k<4; k++)
               v += buf[i+k*257+33];
           hist[m++]=v;
         }

         m = 0;
         for(unsigned i=0; i<256;) {
           unsigned v=0;
           for(unsigned j=0; j<4; j++,i++)
             for(unsigned k=4; k<8; k++)
               v += buf[i+k*257];
           hist_hwf[m++]=v;
         }

         for(unsigned i=0; i<60; i++)
           printf("%04x%c",hist[i],(i&0xf)==0xf?'\n':' ');
         printf("\n  -\n");

         for(unsigned i=0; i<60; i++)
           printf("%04x%c",hist[i]-last[i],(i&0xf)==0xf?'\n':' ');
         printf("\n--\n");

         for(unsigned i=0; i<64; i++)
           printf("%04x%c",hist_hwf[i],(i&0xf)==0xf?'\n':' ');
         printf("\n  -\n");

         for(unsigned i=0; i<64; i++)
           printf("%04x%c",hist_hwf[i]-last_hwf[i],(i&0xf)==0xf?'\n':' ');
         printf("\n----\n");

         unsigned* p = hist;
         hist = last;
         last = p;

         p = hist_hwf;
         hist_hwf = last_hwf;
         last_hwf = p;
       }
     }
   }

   return(0);
}

