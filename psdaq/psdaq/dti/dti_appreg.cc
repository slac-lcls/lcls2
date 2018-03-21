/**
 ** pgpdaq
 **
 **   Manage XPM and DTI to trigger and readout pgpcard (dev03)
 **
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <new>

#include <cpsw_api_builder.h>
#include <cpsw_mmio_dev.h>
#include <cpsw_proto_mod_depack.h>

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -l <lane>                      : Upstream link partition\n");
  printf("         -w <write reg addr>,<value>    : Register write address and value\n");
  printf("         -r <read reg addr>             : Register read address\n");
}

int main(int argc, char** argv) {

  extern char* optarg;

  int c;

  const char* ip  = "10.0.1.103";
  char* endPtr = 0;
  unsigned lane = 0;
  int readReg  = -1;
  int writeReg = -1;
  int writeVal = -1;
  while ( (c=getopt( argc, argv, "a:l:r:w:h")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 'l': lane = strtoul(optarg, NULL, 0); break;
    case 'r': readReg = strtoul(optarg, NULL, 0); break;
    case 'w': writeReg = strtoul(optarg, &endPtr, 0);
      writeVal = strtoul(endPtr+1, NULL, 0);
      break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  NetIODev  root = INetIODev::create("fpga", ip);

  {  // Streaming access
    ProtoStackBuilder bldr = IProtoStackBuilder::create();
    bldr->setSRPVersion              ( IProtoStackBuilder::SRP_UDP_NONE );
    bldr->setUdpPort                 (                 8194+lane );
    bldr->setSRPTimeoutUS            (                 90000 );
    bldr->setSRPRetryCount           (                     5 );
    //    bldr->setSRPMuxVirtualChannel    (                     0 );
    bldr->useDepack                  (                  true );
    bldr->useRssi                    (                  true );
    //    bldr->setTDestMuxTDEST           (                 tdest );

    Field    irq = IField::create("usreg");
    root->addAtAddress( irq, bldr );
  }

  Path path = IPath::create(root);

  Stream strm = IStream::create( path->findByName("usreg") );
  CTimeout         tmo(100000);

  uint8_t buf [1500];
  uint64_t bb;
  unsigned sz;

  CAxisFrameHeader hdr;
  if (writeReg >= 0) {
    hdr.insert(buf, sizeof(buf));
    bb = (writeReg<<2) | 0;
    bb |= uint64_t(writeVal)<<32;
    sz = hdr.getSize();
    for( unsigned i=0; i<sizeof(bb); i++) {
      buf[sz++] = bb & 0xff;
      bb >>= 8;
    }
    hdr.iniTail   ( buf+sz );
    hdr.setTailEOF( buf+sz, true );
    sz += hdr.getTailSize();
    strm->write(buf, sz);
  }

  if (readReg >= 0) {
    hdr.insert(buf, sizeof(buf));
    bb = (readReg<<2) | 1;
    bb |= uint64_t(0)<<32;
    sz = hdr.getSize();
    for( unsigned i=0; i<sizeof(bb); i++) {
      buf[sz++] = bb & 0xff;
      bb >>= 8;
    }
    hdr.iniTail   ( buf+sz );
    hdr.setTailEOF( buf+sz, true );
    sz += hdr.getTailSize();
    strm->write(buf, sz);

    sz = strm->read( buf, sizeof(buf), CTimeout(900000), 0 );
    if (sz == 0) {
      printf("Read timeout\n");
      return -1;
    }

    if (!hdr.parse(buf, sizeof(buf))) {
      printf("bad header\n");
      return -1;
    }

    if (!hdr.getTailEOF(buf + sz - hdr.getTailSize())) {
      printf("no EOF tag\n");
      return -1;
    }

    printf("Frame %u  ", hdr.getFrameNo());
    for( unsigned i=0; i< sz - hdr.getSize() - hdr.getTailSize(); i++)
      printf("%02x", buf[hdr.getSize()+i]);
    printf("\n");
  }

  return 0;
}
