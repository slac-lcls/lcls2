#include "GthEyeScan.hh"

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

using namespace Kcu;

static int row_, column_;

static inline unsigned getf(unsigned i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned getf(const Kcu::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Kcu::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;
  return q;
}

bool GthEyeScan::enabled() const
{
  return (_es_control & (1<<8));
}

void GthEyeScan::enable(bool v)
{
  unsigned control = _es_control;
  if (v)
    control |= (1<<8);
  else
    control &= ~(1<<8);
  _es_control = control;
}

void GthEyeScan::scan(const char* ofile, 
                      unsigned    prescale, 
                      unsigned    xscale,
                      bool        lsparse,
                      bool        lhscan)
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
    setf(_es_control, 0, 1, 10);
  }
  do {
    usleep(1);
  } while ( getf(_es_control_status, 4, 0) != 1 );
  printf("WAIT state\n");
    
  setf(_es_control, 1, 1, 9);  // errdet_en

  setf(_es_control, prescale, 5, 0);

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

  setf(_rx_eyescan_vs, 3, 2, 0); // range
  setf(_rx_eyescan_vs, 0, 1, 9); // ut sign
  setf(_rx_eyescan_vs, 0, 1, 10); // neg_dir
  setf(_es_horz_offset, 0, 12, 4); // zero horz offset

  if (lhscan)
    _hscan(f,xscale,lsparse);
  else
    _vscan(f,xscale,lsparse);

  fclose(f);
}

void GthEyeScan::_vscan(FILE*    f, 
                        unsigned xscale,
                        bool     lsparse)
{
  char stime[200];

  for(int j=-31; j<32; j++) {
    row_ = j;

    time_t t = time(NULL);
    struct tm* tmp = localtime(&t);
    if (tmp)
      strftime(stime, sizeof(stime), "%T", tmp);

    printf("es_horz_offset: %i [%s]\n",j, stime);
    setf(_es_horz_offset, j<<xscale, 12, 4);
    setf(_rx_eyescan_vs, 0, 9, 2); // zero vert offset

    uint64_t sample_count;
    unsigned error_count=-1, error_count_p=-1;

    for(int i=-1; i>=-127; i--) {
      column_ = i;
      setf(_rx_eyescan_vs, i, 9, 2); // vert offset
      run(error_count,sample_count);

      fprintf(f, "%d %d %u %llu\n",
              j, i, 
              error_count,
              (unsigned long long)sample_count);
                
      setf(_es_control, 0, 1, 10); // -> wait

      if (error_count==0 && error_count_p==0 && !lsparse) {
        //          printf("\t%i\n",i);
        break;
      }

      error_count_p=error_count;

      if (lsparse)
        i -= 19;
    }
    setf(_rx_eyescan_vs, 0, 9, 2); // zero vert offset
    error_count_p = -1;
    for(int i=127; i>=0; i--) {
      column_ = i;
      setf(_rx_eyescan_vs, i, 9, 2); // vert offset
      run(error_count,sample_count);

      fprintf(f, "%d %d %u %llu\n",
              j, i, 
              error_count,
              (unsigned long long)sample_count);
                
      setf(_es_control, 0, 1, 10); // -> wait

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
}

void GthEyeScan::_hscan(FILE*    f, 
                        unsigned xscale,
                        bool     lsparse)
{
  char stime[200];

  for(int i=127; i>=-127; i--) {
    column_ = i;

    time_t t = time(NULL);
    struct tm* tmp = localtime(&t);
    if (tmp)
      strftime(stime, sizeof(stime), "%T", tmp);

    printf("es_vert_offset: %i [%s]\n",i, stime);

    setf(_rx_eyescan_vs, i, 8, 2); // vert offset

    uint64_t sample_count;
    unsigned error_count=-1, error_count_p=-1;

    int j;
    for(j=-31; j<32; j++) {
      row_ = j;

      setf(_es_horz_offset, j<<xscale, 12, 4);

      unsigned ec0=0; uint64_t sc0=0;
      setf(_rx_eyescan_vs, 0, 1, 10); // ut sign
      run(ec0,sc0);
      setf(_es_control, 0, 1, 10); // -> wait

      unsigned ec1=0; uint64_t sc1=0;
      setf(_rx_eyescan_vs, 1, 1, 10); // ut sign
      run(ec1,sc1);
      setf(_es_control, 0, 1, 10); // -> wait

      double sc = sqrt(double(sc0)*double(sc1));
      double ec = (double(ec0)*double(sc1)+double(ec1)*double(sc0))/sc;

      error_count  = ec;
      sample_count = sc;

      fprintf(f, "%d %d %u %llu\n",
              j, i, 
              error_count,
              (unsigned long long)sample_count);
                
      if (error_count==0 && error_count_p==0 && !lsparse) {
        //          printf("\t%i\n",i);
        break;
      }

      error_count_p=error_count;

      if (lsparse)
        j += 3;
    }

    if (j<32) {
      int je=j;
      for(j=31; j>je; j--) {
        row_ = j;

        setf(_es_horz_offset, j<<xscale, 12, 4);

        unsigned ec0=0; uint64_t sc0=0;
        setf(_rx_eyescan_vs, 0, 1, 10); // ut sign
        run(ec0,sc0);
        setf(_es_control, 0, 1, 10); // -> wait

        unsigned ec1=0; uint64_t sc1=0;
        setf(_rx_eyescan_vs, 1, 1, 10); // ut sign
        run(ec1,sc1);
        setf(_es_control, 0, 1, 10); // -> wait

        double sc = sqrt(double(sc0)*double(sc1));
        double ec = (double(ec0)*double(sc1)+double(ec1)*double(sc0))/sc;

        error_count  = ec;
        sample_count = sc;

        fprintf(f, "%d %d %u %llu\n",
                j, i, 
                error_count,
                (unsigned long long)sample_count);
                
        setf(_es_control, 0, 1, 10); // -> wait

        if (error_count==0 && error_count_p==0 && !lsparse) {
          //          printf("\t%i\n",i);
          break;
        }

        error_count_p=error_count;

        if (lsparse)
          j += 3;
      }
    }

    if (lsparse)
      i -= 19;
  }
}

void GthEyeScan::run(unsigned& error_count,
                     uint64_t& sample_count)
{
  setf(_es_control, 1, 1, 10); // -> run
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
  sample_count <<= (1 + getf(_es_control,5,0));
}            

void GthEyeScan::progress(unsigned& row,
                          unsigned& col)
{
  row = row_;
  col = column_;
}
