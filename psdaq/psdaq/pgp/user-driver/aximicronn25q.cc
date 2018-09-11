#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "aximicronn25q.hh"
#include "mcsfile.hh"

enum { READ_3BYTE_CMD    = 0x03<<16 };
enum { READ_4BYTE_CMD    = 0x13<<16 };
enum { FLAG_STATUS_REG   = 0x70<<16 };
enum { FLAG_STATUS_RDY   = 0x80 };
enum { WRITE_ENABLE_CMD  = 0x06<<16 };
enum { WRITE_DISABLE_CMD = 0x04<<16 };
enum { ADDR_ENTER_CMD    = 0xB7<<16 };
enum { ADDR_EXIT_CMD     = 0xE9<<16 };
enum { ERASE_CMD         = 0xD8<<16 };
enum { WRITE_CMD         = 0x02<<16 };
enum { STATUS_REG_WR_CMD = 0x01<<16 };
enum { STATUS_REG_RD_CMD = 0x05<<16 };
enum { DEV_ID_RD_CMD     = 0x9F<<16 };
enum { WRITE_NONVOLATILE_CONFIG  = 0xB1<<16 };
enum { WRITE_VOLATILE_CONFIG     = 0x81<<16 };
enum { READ_NONVOLATILE_CONFIG   = 0xB5<<16 };
enum { READ_VOLATILE_CONFIG      = 0x85<<16 };
enum { DEFAULT_3BYTE_CONFIG = 0xFFFF };
enum { DEFAULT_4BYTE_CONFIG = 0xFFFE };
enum { READ_MASK   = 0 };
enum { WRITE_MASK  = 0x80000000 };
enum { VERIFY_MASK = 0x40000000 };

static bool _verbose = false;

//  Assume memory mapped
class Device {
public:
  Device(char*    base,
         unsigned chunk=256) : _p(base), _data(new char[chunk]) {}
  ~Device() { delete[] _data; }
public:
  void      rawWrite(unsigned offset, unsigned v) 
  { if (_verbose) printf("rawWrite [%p]=0x%x\n", _p+offset,v);
    *reinterpret_cast<volatile uint32_t*>(_p+offset) = v; }
  volatile unsigned  rawRead (unsigned offset)
  { if (_verbose) printf("rawRead [%p]\n",_p+offset);
    return *reinterpret_cast<volatile uint32_t*>(_p+offset); }
  void      rawWrite(unsigned offset, volatile unsigned* arg1)
  { for(unsigned i=0; i<64; i++) rawWrite(offset+4*i,arg1[i]); }
  volatile uint32_t* rawRead (unsigned offset, unsigned nword)
  { uint32_t* u = reinterpret_cast<uint32_t*>(_data);
    for(unsigned i=0; i<nword; i++) u[i] = rawRead(offset+4*i);
    return u; }
private:
  volatile char* _p;
  char* _data;
};

class AxiMicronN25Q::PrivateData {
public:
  bool     _addrMode; // true=32b mode
  McsFile* mcs;
  Device   dev;
public:
  PrivateData(char* base,
              const char* fname) : 
    _addrMode(false), mcs(new McsFile(fname)), dev(base) {}
  ~PrivateData() { delete mcs; }
public:
  void eraseProm()
  {
    const unsigned ERASE_SIZE = 0x10000;
    unsigned address = mcs->startAddr();
    const unsigned size = mcs->write_size();
    unsigned report = address + size/10;
    while(address < mcs->endAddr()) {
      eraseCmd(address);
      address += ERASE_SIZE;
      if (address > report) {
        printf("Erase %d%% complete\n", 100*(address-mcs->startAddr())/size);
        report += size/10;
      }
    }
  }
  void writeProm()
  {
    unsigned wordCnt=0, byteCnt=0, wrd=0;
    unsigned addr,dummy,data;
    volatile uint32_t dataArray[64];
    const unsigned write_size = mcs->write_size();
    unsigned report = write_size/10;
    unsigned nPrint=0;
    for(unsigned i=0; i<write_size; i++) {
      if (byteCnt==0) {
        if (wordCnt==0)
          mcs->entry(i,addr,data);
        else
          mcs->entry(i,dummy,data);
        wrd = (data&0xff) << 24;
        ++byteCnt;
      }
      else {
        mcs->entry(i,dummy,data);
        wrd |= (data&0xff) << (8*(3-byteCnt));

        if (++byteCnt==4) {
          byteCnt=0;
          dataArray[wordCnt++] = wrd;
          if (wordCnt==64) {
            wordCnt=0;
            setDataReg(dataArray);
            writeCmd(addr);
            if (nPrint) {
              nPrint--;
              printf("Write [addr=%x]:",addr);
              for(unsigned j=0; j<64; j++)
                printf(" %08x", dataArray[j]);
              printf("\n");
            }
          }
        }
      }
      if (i > report) {
        printf("WriteProm %d%% complete\n", 100*i/write_size);
        report += write_size/10;
      }
    }
    printf("Final write: wordCnt %u, byteCnt %u, wrd %x\n",
           wordCnt, byteCnt, wrd);
  }
  void verifyProm()
  {
    waitForFlashReady();
    unsigned wordCnt=0, byteCnt=0, data,addr;
    volatile unsigned* dataArray;
    const unsigned read_size = mcs->read_size();
    unsigned report = read_size/10;
    unsigned nPrint=0;
    bool lFail=false;
    unsigned nFail=0;
    for (unsigned i=0; i<read_size; i++) {
      if (!byteCnt && !wordCnt) {
        mcs->entry(i,addr,data);
        readCmd(addr);
        dataArray = getDataReg();
        if (nPrint) {
          nPrint--;
          printf("Read [addr=%x]:",addr);
          for(unsigned j=0; j<64; j++)
            printf(" %08x", dataArray[j]);
          printf("\n");
        }
        lFail=false;
      }
      mcs->entry(i,addr,data);
      unsigned prom = ((dataArray[wordCnt] >> (8*(3-byteCnt))) & 0xff);
      if (data != prom && !lFail) {
        printf("VerifyProm failed!\n");
        printf("Addr = 0x%x:  WordCnt %x, ByteCnt %x, MCS = 0x%x != PROM = 0x%x\n",
               addr,wordCnt,byteCnt,data,prom);
        lFail=true;
        if (++nFail==20)
          exit(1);
      }
      if (++byteCnt==4) {
        byteCnt=0;
        if (++wordCnt==64)
          wordCnt=0;
      }
      if (i > report) {
        printf("Verify %d%% complete\n",100*i/read_size);
        report += read_size/10;
      }
    }
  }
public:
  void resetFlash()
  {
    setCmdReg(WRITE_MASK|(0x66<<16));
    usleep(1000);
    setCmdReg(WRITE_MASK|(0x99<<16));
    usleep(1000);
    setModeReg();
    if (_addrMode) {
      setCmd(WRITE_MASK|ADDR_ENTER_CMD);
      setAddrReg(DEFAULT_4BYTE_CONFIG<<16);
    }
    else {
      setCmd(WRITE_MASK|ADDR_EXIT_CMD);
      setAddrReg(DEFAULT_3BYTE_CONFIG<<8);
    }
    usleep(1000);
    setCmd(WRITE_MASK|WRITE_NONVOLATILE_CONFIG|0x2);
    setCmd(WRITE_MASK|WRITE_VOLATILE_CONFIG|0x2);
  }
  void eraseCmd(unsigned address)
  {
    setAddrReg(address);
    setCmd(WRITE_MASK | ERASE_CMD | (_addrMode ? 0x4:0x3));
  }
  void writeCmd(unsigned address)
  {
    setAddrReg(address);
    setCmd(WRITE_MASK|WRITE_CMD|(_addrMode?0x104:0x103));
  }
  void readCmd(unsigned address)
  {
    setAddrReg(address);
    setCmd(READ_MASK|(_addrMode?(READ_4BYTE_CMD|0x104):(READ_3BYTE_CMD|0x103)));
  }
  void setCmd(unsigned value)
  {
    if (value&WRITE_MASK) {
      waitForFlashReady();
      setCmdReg(WRITE_MASK|WRITE_ENABLE_CMD);
    }
    setCmdReg(value);
  }
  void waitForFlashReady()
  {
    while(1) {
      setCmdReg(READ_MASK|FLAG_STATUS_REG|0x1);
      volatile unsigned status = getCmdReg()&0xff;
      if (( status & FLAG_STATUS_RDY)) 
        break;
    }
  }
  void setModeReg()            { dev.rawWrite(0x04,_addrMode?0x1:0x0); }
  void setAddrReg(volatile unsigned v)  { dev.rawWrite(0x08,v); }
  void setCmdReg (volatile unsigned v)  { dev.rawWrite(0x0c,v); }
  void setDataReg(volatile unsigned* v) { dev.rawWrite(0x200,v); }
  volatile unsigned  getCmdReg () { return dev.rawRead(0x0c); }
  volatile unsigned* getDataReg() { return dev.rawRead(0x200,64); }

  void setPromStatusReg(unsigned value)
  {
    setAddrReg((value&0xff)<<(_addrMode?24:16));
    setCmd(WRITE_MASK|STATUS_REG_WR_CMD|0x1);
  }
  unsigned getPromStatusReg()
  {
    setCmd(READ_MASK|STATUS_REG_RD_CMD|0x1);
    return getCmdReg()&0xff;
  }
  unsigned getFlagStatusReg()
  {
    setCmd(READ_MASK|FLAG_STATUS_REG|0x1);
    return getCmdReg()&0xff;
  }
  unsigned getPromConfigReg()
  {
    setCmd(READ_MASK|DEV_ID_RD_CMD|0x1);
    return getCmdReg()&0xff;
  }
  unsigned getManufacturerId()
  {
    setCmd(READ_MASK|DEV_ID_RD_CMD|0x1);
    return getCmdReg()&0xff;
  }
  unsigned getManufacturerType()
  {
    setCmd(READ_MASK|DEV_ID_RD_CMD|0x2);
    return getCmdReg()&0xff;
  }
  unsigned getManufacturerCapacity()
  {
    setCmd(READ_MASK|DEV_ID_RD_CMD|0x3);
    return getCmdReg()&0xff;
  }
};

AxiMicronN25Q::AxiMicronN25Q(char*       base,
                             const char* fname) :  
  _private(new PrivateData(base,fname)) {}

AxiMicronN25Q::~AxiMicronN25Q() { delete _private; }

void AxiMicronN25Q::load()
{
  _private->resetFlash();

  printf("MicronN25Q Manufacturer ID Code  = %x\n",_private->getManufacturerId());
  printf("MicronN25Q Manufacturer Type     = %x\n",_private->getManufacturerType());
  printf("MicronN25Q Manufacturer Capacity = %x\n",_private->getManufacturerCapacity());
  printf("MicronN25Q Status Register       = %x\n",_private->getPromStatusReg());
  printf("MicronN25Q Flag Status Register  = %x\n",_private->getFlagStatusReg());
  printf("MicronN25Q Volatile Config Reg   = %x\n",_private->getPromConfigReg());

  _private->eraseProm();

  _private->writeProm();
}

void AxiMicronN25Q::verify()
{
  printf("MicronN25Q Manufacturer ID Code  = %x\n",_private->getManufacturerId());
  printf("MicronN25Q Manufacturer Type     = %x\n",_private->getManufacturerType());
  printf("MicronN25Q Manufacturer Capacity = %x\n",_private->getManufacturerCapacity());
  printf("MicronN25Q Status Register       = %x\n",_private->getPromStatusReg());
  printf("MicronN25Q Flag Status Register  = %x\n",_private->getFlagStatusReg());
  printf("MicronN25Q Volatile Config Reg   = %x\n",_private->getPromConfigReg());
  _private->verifyProm();
}
