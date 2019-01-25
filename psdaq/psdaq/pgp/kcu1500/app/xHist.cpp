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

int main (int argc, char **argv) {
   uint8_t       mask[DMA_MASK_SIZE];
   int32_t       ret;
   int32_t       s;
   uint32_t      rxFlags;
   uint32_t      dmaDest=0;

   const char* dev = "/dev/datadev_1";
   extern char* optarg;
   char c;
   while((c = getopt(argc,argv,"d:"))!=EOF) {
     switch(c) {
     case 'd': dev = optarg; break;
     default: usage(argv[0]); exit(1);
     }
   }

   const unsigned buf_size = 0x400000;
   uint32_t* buf = new uint32_t[buf_size>>2];

   dmaInitMaskBytes(mask);
   dmaAddMaskBytes((uint8_t*)mask,(4<<8));

   if ( (s = open(dev, O_RDWR)) <= 0 ) {
      printf("Error opening %s\n",dev);
      return(1);
   }

   dmaSetMaskBytes(s,mask);

   unsigned prescale=0;
   unsigned* last = new unsigned[64];
   unsigned* hist = new unsigned[64];

   while(1) {

     // DMA Read
     do {
       ret = dmaRead(s,buf,buf_size,&rxFlags,NULL,&dmaDest);
     } while(ret == 0);
     if (++prescale < 100) 
       continue;

     prescale = 0;

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

     for(unsigned i=0; i<60; i++)
       printf("%04x%c",hist[i],(i&0xf)==0xf?'\n':' ');
     printf("\n  -\n");

     for(unsigned i=0; i<60; i++)
       printf("%04x%c",hist[i]-last[i],(i&0xf)==0xf?'\n':' ');
     printf("\n--\n");

     unsigned* p = hist;
     hist = last;
     last = p;
   }

   return(0);
}

