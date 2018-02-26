
#include <sys/types.h>
#include <sys/mman.h>

#include <linux/types.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <semaphore.h>

#include "../include/PgpCardMod.h"
#include "../include/PgpCardReg.h"

#define PAGE_SIZE 4096

namespace Pgp {
  class Reg {
  public:
    Reg& operator=(const unsigned);
    operator unsigned() const;
  public:
    static void init(void*);
  private:
    uint32_t _reserved;
  };

  class GtpEyeScan {
  public:
    bool enabled() const;
    void enable(bool);
    void scan(const char* ofile,
              unsigned    prescale=0,
              unsigned    xscale=0,
              bool        lsparse=false);
    void run(unsigned& error_count,
             uint64_t& sample_count);
    static void progress(unsigned& row,
                         unsigned& col);
  public:
    uint32_t _reserved_2c[0x2c];
    Reg      _es_qualifier[5];
    Reg      _es_qual_mask[5];
    Reg      _es_sdata_mask[5];
    Reg      _es_vert_offset; // [8:0], prescale [15:11]
    Reg      _es_horz_offset; // [11:0]
    Reg      _es_control;  // [5:0] control, [9] errdet_en, [8], eye_scan_en
    uint32_t _reserved_91[0x91-0x3e];
    Reg      _es_clkph_sel; // [15]
    uint32_t _reserved_a6[0xA6-0x92];
    Reg      _es_pma_cfg; // [9:0]
    uint32_t _reserved_151[0x151-0xA7];
    Reg      _es_error_count;
    Reg      _es_sample_count;
    Reg      _es_control_status;
    Reg      _es_rdata[5];
    Reg      _es_sdata[5];
    uint32_t _reserved_200[0x200-0x15E];
    uint32_t _reserved_extra[0x200];
  };
};

using namespace Pgp;

static sem_t _sem;
static uint32_t* _drpReg;
static bool lverbose = false;

void Reg::init(void* m)
{
  sem_init(&_sem,0,1);

  GtpEyeScan* p = new(0) GtpEyeScan;
  printf("es_qual @%p\n",&p->_es_qualifier[0]);
  printf("es_vert @%p\n",&p->_es_vert_offset);
  printf("es_cntl @%p\n",&p->_es_control);
  printf("es_errc @%p\n",&p->_es_error_count);
  printf("es_smpc @%p\n",&p->_es_sample_count);

  _drpReg = &reinterpret_cast<uint32_t*>(m)[0];
  for(unsigned i=0; i<8; i++)
    printf("drpReg [%p] = %08x\n",
           _drpReg, *_drpReg);

  _drpReg = &reinterpret_cast<uint32_t*>(m)[0x22];
  for(unsigned i=0; i<8; i++)
    printf("drpReg [%p] = %08x\n",
           _drpReg, *_drpReg);
}

static unsigned wait_for_ready()
{
  unsigned v,vo=0;
  while(1) {
    v = *_drpReg;
    if (lverbose && v!=vo) {
      vo=v;
      printf("\tread %08x\n",v);
    }
    if (v&1) break;
    usleep(100);
  }
  return v;
}

Reg& Reg::operator=(const unsigned r)
{
  sem_wait(&_sem);

  unsigned addr = reinterpret_cast<const char*>(this) - (const char*)0;
  if (lverbose)
    printf("addr %x\n",addr);

  unsigned cmd = ((r&0xffff)<<16) | (addr&0xfffc) | (1<<1);
  *_drpReg = cmd;
  if (lverbose)
    printf("\twrote %08x\n",cmd);

  wait_for_ready();

  sem_post(&_sem);
  return *this;
}

Reg::operator unsigned() const
{
  sem_wait(&_sem);

  unsigned addr = reinterpret_cast<const char*>(this) - (const char*)0;
  if (lverbose)
    printf("addr %x\n",addr);

  unsigned cmd = addr & 0xfffc;
  *_drpReg = cmd;
  if (lverbose)
    printf("\twrote %08x\n",cmd);

  unsigned v = wait_for_ready();
  
  sem_post(&_sem);
  return v>>16;
}

static int row_, column_;

static inline unsigned getf(unsigned i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned getf(const Pgp::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Pgp::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;
  return q;
}

bool GtpEyeScan::enabled() const
{
  return (_es_control & (1<<8));
}

void GtpEyeScan::enable(bool v)
{
  printf("enable %c\n",v?'T':'F');
  unsigned control = _es_control;
  if (v)
    control |= (1<<8);
  else
    control &= ~(1<<8);
  _es_control = control;
}

void GtpEyeScan::scan(const char* ofile, 
                      unsigned    prescale, 
                      unsigned    xscale,
                      bool        lsparse)
{
  FILE* f = fopen(ofile,"w");

  unsigned status = _es_control_status;
  printf("eyescan status: %04x\n",status);

  for(unsigned i=0; i<5; i++) {
    unsigned data = getf(_es_sdata_mask[i], 16, 0);
    unsigned qual = getf(_es_qual_mask [i], 16, 0);
    printf("data qual : %x %x\n",data,qual);
  }

  if ((status & 0xe) != 0) {
    printf("Forcing to WAIT state\n");
    setf(_es_control, 0, 1, 0);
  }
  do {
    usleep(1);
  } while ( getf(_es_control_status, 4, 0) != 1 );
  printf("WAIT state\n");
    
  setf(_es_control, 1, 1, 9);  // errdet_en

  setf(_es_vert_offset, prescale, 5, 11);

  setf(_es_sdata_mask[0], 0xffff, 16, 0);
  setf(_es_sdata_mask[1], 0xffff, 16, 0);
  setf(_es_sdata_mask[2], 0xff00, 16, 0);
  setf(_es_sdata_mask[3], 0x000f, 16, 0);
  setf(_es_sdata_mask[4], 0xffff, 16, 0);
  for(unsigned i=0; i<5; i++)
    setf(_es_qual_mask[i], 0xffff, 16, 0);

  for(unsigned i=0; i<5; i++) {
    unsigned data = getf(_es_sdata_mask[i], 16, 0);
    unsigned qual = getf(_es_qual_mask [i], 16, 0);
    printf("data qual : %x %x\n",data,qual);
  }

  // setf(_rx_eyescan_vs, 3, 2, 0); // range
  // setf(_rx_eyescan_vs, 0, 1, 9); // ut sign
  // setf(_rx_eyescan_vs, 0, 1, 10); // neg_dir
  setf(_es_horz_offset, 0, 12, 0); // zero horz offset

  char stime[200];

  for(int j=-31; j<32; j++) {
    row_ = j;

    time_t t = time(NULL);
    struct tm* tmp = localtime(&t);
    if (tmp)
      strftime(stime, sizeof(stime), "%T", tmp);

    printf("es_horz_offset: %i [%s]\n",j, stime);
    setf(_es_horz_offset, j<<xscale, 12, 0);
    setf(_es_vert_offset, 0, 9, 0); // zero vert offset

    uint64_t sample_count;
    unsigned error_count=-1, error_count_p=-1;

    for(int i=-1; i>=-127; i--) {
      column_ = i;
      setf(_es_vert_offset, i, 9, 0); // vert offset
      run(error_count,sample_count);

      fprintf(f, "%d %d %u %llu\n",
              j, i, 
              error_count,
              (unsigned long long)sample_count);
                
      setf(_es_control, 0, 1, 0); // -> wait

      if (error_count==0 && error_count_p==0 && !lsparse) {
        //          printf("\t%i\n",i);
        break;
      }

      error_count_p=error_count;

      if (lsparse)
        i -= 19;
    }
    setf(_es_vert_offset, 0, 9, 0); // zero vert offset
    error_count_p = -1;
    for(int i=127; i>=0; i--) {
      column_ = i;
      setf(_es_vert_offset, i, 9, 0); // vert offset
      run(error_count,sample_count);

      fprintf(f, "%d %d %u %llu\n",
              j, i, 
              error_count,
              (unsigned long long)sample_count);
                
      setf(_es_control, 0, 1, 0); // -> wait

      if (error_count==0 && error_count_p==0 && !lsparse) {
        //          printf("\t%i\n",i);
        break;
      }

      error_count_p=error_count;

      if (lsparse)
        i -= 19;
    }
    if (lsparse)
      j += 3;
  }
  fclose(f);
}

void GtpEyeScan::run(unsigned& error_count,
                     uint64_t& sample_count)
{
  setf(_es_control, 1, 1, 0); // -> run
  while(1) {
    unsigned nwait=0;
    do {
      usleep(100);
      nwait++;
    } while(getf(_es_control_status,1,0)==0 and nwait < 1000);
    if (getf(_es_control_status,3,1)==2)
      break;
    //        printf("\tstate : %x\n", getf(_es_control_status,3,1));
  }
  error_count  = _es_error_count;
  sample_count = _es_sample_count;
  sample_count <<= (1 + getf(_es_vert_offset,5,11));
}            

void GtpEyeScan::progress(unsigned& row,
                          unsigned& col)
{
  row = row_;
  col = column_;
}


static GtpEyeScan* gtp;
static const char* outfile = "eyescan.dat";
static unsigned prescale = 0;
static bool lsparse = false;

void* scan_routine(void* arg)
{
  unsigned lane = *(unsigned*)arg;
  printf("Start lane %u\n",lane);

  char ofile[64];
  sprintf(ofile,"%s.%u",outfile,lane);

  printf("gtp[%u] @ %p\n",lane,&gtp[lane]);

  //  lverbose=true;
  gtp[lane].enable(true);
  if (gtp[lane].enabled()) 
    gtp[lane].scan(ofile, prescale, 0, lsparse);
  else
    printf("enable failed\n");

  return 0;
}

void showUsage(const char* p) {
  printf("Usage: %s [options]\n", p);
  printf("Options:\n"
         "\t-P <dev>      Use pgpcard <dev> (integer)\n"
         "\t-L <lanes>    Bit mask of lanes\n"
         "\t-f <filename> Output file\n"
         "\t-p <prescale> Prescale exponent\n"
         "\t-s            Sparse mode\n");
}

using std::cout;
using std::endl;
using std::dec;
using std::hex;

int main (int argc, char **argv) {
  int           fd;
  int           ret;
  unsigned      idev=0;
  unsigned      lanes=1;
  char dev[64];

  int c;

  while((c=getopt(argc,argv,"P:L:f:p:s")) != EOF) {
    switch(c) {
    case 'P': idev   = strtoul(optarg,NULL,0); break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    case 'f': outfile = optarg; break;
    case 'p': prescale = strtoul(optarg,NULL,0); break;
    case 's': lsparse = true; break;
    default:
      showUsage(argv[0]); return 0;
    }
  }

  sprintf(dev,"/dev/pgpcardG3_%u_0",idev);
  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    perror(dev);
    return(1);
  }


  void volatile *mapStart;

  // Map the PCIe device from Kernel to Userspace
  mapStart = (void volatile *)mmap(NULL, PAGE_SIZE, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   
  if(mapStart == MAP_FAILED){
    cout << "Error: mmap() = " << dec << mapStart << endl;
    close(fd);
    return(1);   
  }

  Reg::init((void*)mapStart);

  gtp = new (0)GtpEyeScan;

  pthread_t tid[8];
  unsigned lane[8];

  for(unsigned i=0; i<8 ;i++) {
    if (lanes & (1<<i)) {
      pthread_attr_t tattr;
      pthread_attr_init(&tattr);
      lane[i] = i;
      if (pthread_create(&tid[i], &tattr, &scan_routine, &lane[i]))
        perror("Error creating scan thread");
    }
  } 

  void* retval;
  for(unsigned i=0; i<8; i++)
    if (lanes & (1<<i))
      pthread_join(tid[i], &retval);

  return 0;
}
