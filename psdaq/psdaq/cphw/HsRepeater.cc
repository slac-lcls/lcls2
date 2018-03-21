#include "psdaq/cphw/HsRepeater.hh"

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <new>

using namespace Pds::Cphw;

// From DS125BR401 datasheet table 2:
static const uint8_t eq[] = { 0x00, 0x01, 0x02, 0x03, 0x07, 0x15, 0x0B, 0x0F,
                              0x55, 0x1f, 0x2f, 0x3f, 0xaa, 0x7f, 0xbf, 0xff };


static unsigned fmco(unsigned short v);


static inline unsigned getf(const Pds::Cphw::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Pds::Cphw::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;
  return q;
}


HsRepeater::HsRepeater() {}

bool    HsChannel::sigDetPreset()       const { return getf(_sigDetFrc,    1, 1); }
void    HsChannel::sigDetPreset(bool v)       {        setf(_sigDetFrc, v, 1, 1); }
bool    HsChannel::sigDetReset()        const { return getf(_sigDetFrc,    1, 2); }
void    HsChannel::sigDetReset(bool v)        {        setf(_sigDetFrc, v, 1, 2); }

uint8_t HsChannel::rxdet()              const { return getf(_rxDet,        2, 2); }
void    HsChannel::rxdet(uint8_t v)           {        setf(_rxDet,     v, 2, 2); }
bool    HsChannel::idle_sel()           const { return getf(_rxDet,        1, 4); }
void    HsChannel::idle_sel(bool v)           {        setf(_rxDet,     v, 1, 4); }
bool    HsChannel::idle_auto()          const { return getf(_rxDet,        1, 5); }
void    HsChannel::idle_auto(bool v)          {        setf(_rxDet,     v, 1, 5); }

void    HsChannel::eqCtl(unsigned i)          { _eqCtl = eq[(i-1)&0xf];           }

uint8_t HsChannel::vodCtl()             const { return getf(_vod,          3, 0); }
void    HsChannel::vodCtl(uint8_t v)          {        setf(_vod,       v, 3, 0); }
bool    HsChannel::mode_sel()           const { return getf(_vod,          1, 6); }
void    HsChannel::mode_sel(bool v)           {        setf(_vod,       v, 1, 6); }
bool    HsChannel::shortCircProt()      const { return getf(_vod,          1, 7); }
void    HsChannel::shortCircProt(bool v)      {        setf(_vod,       v, 1, 7); }

uint8_t HsChannel::demCtl()             const { return getf(_dem,          3, 0); }
void    HsChannel::demCtl(uint8_t v)          {        setf(_dem,       v, 3, 0); }
uint8_t HsChannel::mode_detStatus()     const { return getf(_dem,          2, 5); }
void    HsChannel::mode_detStatus(uint8_t v)  {        setf(_dem,       v, 2, 5); }
bool    HsChannel::rxdetStatus()        const { return getf(_dem,          1, 7); }
void    HsChannel::rxdetStatus(bool v)        {        setf(_dem,       v, 1, 7); }

uint8_t HsChannel::idleThd()            const { return getf(_idleThrsh,    2, 0); }
void    HsChannel::idleThd(uint8_t v)         {        setf(_idleThrsh, v, 2, 0); }
uint8_t HsChannel::idleTha()            const { return getf(_idleThrsh,    2, 2); }
void    HsChannel::idleTha(uint8_t v)         {        setf(_idleThrsh, v, 2, 2); }


bool    HsRepeater::eepromReadDone()    const { return getf(_devAddr,      1, 2); }
uint8_t HsRepeater::addressBits()       const { return getf(_devAddr,      4, 3); }

bool    HsRepeater::pwdnChan(uint8_t i) const { return getf(_pwdnChans,    1, i & 0x7); }
void    HsRepeater::pwdnChan(uint8_t i,
                             bool v)          {        setf(_pwdnChans, v, 1, i & 0x7); }

bool    HsRepeater::pwdnOverride()      const { return getf(_ovrPwdn,      1, 0); }
void    HsRepeater::pwdnOverride(bool v)      {        setf(_ovrPwdn,   v, 1, 0); }
uint8_t HsRepeater::lpbkCtl()           const { return getf(_ovrPwdn,      2, 4); }
void    HsRepeater::lpbkCtl(uint8_t v)        {        setf(_ovrPwdn,   v, 2, 4); }

bool    HsRepeater::smbusEnable()       const { return getf(_slvRegCtl,    1, 3); }
void    HsRepeater::smbusEnable(bool v)       {        setf(_slvRegCtl, v, 1, 3); }

void    HsRepeater::resetSmbus()              {        setf(_digRstCtl, 1, 1, 5); }
void    HsRepeater::resetRegs()               {        setf(_digRstCtl, 1, 1, 6); }

bool    HsRepeater::modePin()           const { return getf(_ovrPinCtl,    1, 2); }
void    HsRepeater::modePin(bool v)           {        setf(_ovrPinCtl, v, 1, 2); }
bool    HsRepeater::rxdetPin()          const { return getf(_ovrPinCtl,    1, 3); }
void    HsRepeater::rxdetPin(bool v)          {        setf(_ovrPinCtl, v, 1, 3); }
bool    HsRepeater::idleCtl()           const { return getf(_ovrPinCtl,    1, 4); }
void    HsRepeater::idleCtl(bool v)           {        setf(_ovrPinCtl, v, 1, 4); }
bool    HsRepeater::sd_thPin()          const { return getf(_ovrPinCtl,    1, 6); }
void    HsRepeater::sd_thPin(bool v)          {        setf(_ovrPinCtl, v, 1, 6); }

bool    HsRepeater::sigDet(uint8_t i)   const { return getf(_sigDetMon,    1, i & 0x7); }

uint8_t HsRepeater::reducedGain()       const { return getf(_sigDetCtl,    2, 0); }
void    HsRepeater::reducedGain(uint8_t v)    {        setf(_sigDetCtl, v, 2, 0); }
uint8_t HsRepeater::fastIdle()          const { return getf(_sigDetCtl,    2, 2); }
void    HsRepeater::fastIdle(uint8_t v)       {        setf(_sigDetCtl, v, 2, 2); }
uint8_t HsRepeater::highIdle()          const { return getf(_sigDetCtl,    2, 4); }
void    HsRepeater::highIdle(uint8_t v)       {        setf(_sigDetCtl, v, 2, 4); }

uint8_t HsRepeater::devVer()            const { return getf(_devID,        5, 0); }
uint8_t HsRepeater::devId()             const { return getf(_devID,        3, 5); }

void HsChannel::dump() const
{
  printf("%-10s:       %p\n", "CH Base", this);

#define PRT(r) printf("%10.10s: 0x%02x", #r, unsigned(r))
  PRT(_sigDetFrc);  printf("  PST %d  RST %d\n", sigDetPreset(), sigDetReset());
  PRT(_rxDet);      printf("  RXDET %02x  IDLE_SEL %d  IDLE_AUTO %d\n",
                           rxdet(), idle_sel(), idle_auto());
  PRT(_eqCtl);      printf("\n");
  PRT(_vod);        printf("  Ctl %01x  MODE_SEL %d  Short Circuit Protect %d\n",
                           vodCtl(), mode_sel(), shortCircProt());
  PRT(_dem);        printf("  Ctl %01x  MODE_DET %d  RXDET %d\n",
                           demCtl(), mode_detStatus(), rxdetStatus());
  PRT(_idleThrsh);  printf("  Thd %01x  Tha %01x\n", idleThd(), idleTha());
#undef PRT
}

void HsChannel::save(FILE* f) const
{
#define PUTR(reg) fprintf(f,"%02x\t%s\n",unsigned(reg),#reg)
  PUTR(_sigDetFrc);
  PUTR(_rxDet);
  PUTR(_eqCtl);
  PUTR(_vod);
  PUTR(_dem);
  PUTR(_idleThrsh);
#undef PUTR
}

void HsChannel::load(FILE* f)
{
#define GETR(reg) { unsigned v; fscanf(f,"%02x\t" #reg "\n",&v); reg=v; }
  GETR(_sigDetFrc);
  GETR(_rxDet);
  GETR(_eqCtl);
  GETR(_vod);
  GETR(_dem);
  GETR(_idleThrsh);
#undef GETR
}

void HsRepeater::init()
{
  smbusEnable(true);
  for(unsigned i=0; i<NChannels; i++) {
    _chA[i].sigDetPreset(1);
    _chA[i].mode_sel(0);
    _chA[i]._eqCtl = 0;
    _chB[i].sigDetPreset(1);
    _chB[i].mode_sel(0);
    _chB[i]._eqCtl = 0;
  }
}

void HsRepeater::dump(unsigned channels) const
{
  printf("%-10s:       %p\n", "Base", this);

#define PRT(r) printf("%10.10s: 0x%02x", #r, unsigned(r))
  PRT(_devID);      printf("  %d.%d\n", devVer(), devId());
  PRT(_devAddr);    printf("  EEPROM Rd done %d  AD %01x\n",
                           eepromReadDone(), addressBits());
  PRT(_pwdnChans);  printf("  0:%d  1:%d  2:%d  3:%d  4:%d  5:%d  6:%d  7:%d\n",
                           pwdnChan(0), pwdnChan(1), pwdnChan(2), pwdnChan(3),
                           pwdnChan(4), pwdnChan(5), pwdnChan(6), pwdnChan(7));
  PRT(_ovrPwdn);    printf("  PWDN %d  LPBK %d\n", pwdnOverride(), lpbkCtl());
  PRT(_slvRegCtl);  printf("  SMBus Enb %d\n", smbusEnable());
  PRT(_digRstCtl);  printf("\n");
  PRT(_ovrPinCtl);  printf("  MODE %d  RXDET %d  IDLE %d  SD_TH %d\n",
                           modePin(), rxdetPin(), idleCtl(), sd_thPin());
  PRT(_sigDetMon);  printf("  0:%d  1:%d  2:%d  3:%d  4:%d  5:%d  6:%d  7:%d\n",
                           sigDet(0), sigDet(1), sigDet(2), sigDet(3),
                           sigDet(4), sigDet(5), sigDet(6), sigDet(7));
  PRT(_sigDetCtl);  printf("  Reduced Gain %d  Fast IDLE %d  High IDLE %d\n",
                          reducedGain(), fastIdle(), highIdle());

  unsigned chans = channels & ((1 << NChannels)-1);
  const HsChannel* hsc = _chB;
  for (unsigned chan = 0; chans; ++chan, ++hsc) {
    if (chans & (1<<chan)) {
      hsc->dump();
      chans &= ~(1<<chan);
    }
  }
  chans = channels & ((1 << NChannels)-1);
  hsc = _chA;
  for (unsigned chan = 0; chans; ++chan, ++hsc) {
    if (chans & (1<<chan)) {
      hsc->dump();
      chans &= ~(1<<chan);
    }
  }
#undef PRT
}

void HsRepeater::save(FILE* f) const
{
#define PUTR(reg) fprintf(f,"%02x\t%s\n",unsigned(reg),#reg)
  PUTR(_pwdnChans);
  PUTR(_ovrPwdn  );
  PUTR(_slvRegCtl);
  PUTR(_digRstCtl);
  PUTR(_sigDetCtl);
#undef PUTR
  for(unsigned i=0; i<NChannels; i++)
    _chA[i].save(f);
  for(unsigned i=0; i<NChannels; i++)
    _chB[i].save(f);
}

void HsRepeater::load(FILE* f)
{
  smbusEnable(true);
#define GETR(reg) { unsigned v; fscanf(f,"%02x\t" #reg "\n",&v); reg=v; }
  GETR(_pwdnChans);
  GETR(_ovrPwdn  );
  GETR(_slvRegCtl);
  GETR(_digRstCtl);
  GETR(_sigDetCtl);
#undef GETR
  for(unsigned i=0; i<NChannels; i++)
    _chA[i].load(f);
  for(unsigned i=0; i<NChannels; i++)
    _chB[i].load(f);
}

unsigned HsRepeater::scanLink(unsigned chan, bool chA, bool chB, MeasFn measFn, void* arg)
{
  unsigned errs[16];
  memset(errs, 0, sizeof(errs));

  smbusEnable(true);
  for (unsigned i = 0; i < sizeof(errs)/sizeof(*errs); ++i)
  {
    unsigned eqVal = eq[(10 + i) & 0xf]; // Start from the default position
    if (chB)
      _chB[chan]._eqCtl = eqVal;
    if (chA)
      _chA[chan]._eqCtl = eqVal;
    errs[(10 + i) & 0xf] = measFn(arg);
  }
  smbusEnable(false);

  unsigned eMin = UINT_MAX;
  for (unsigned i = 0; i < sizeof(errs)/sizeof(*errs); ++i)
  {
    if (errs[i] < eMin)  eMin = errs[i];
  }
  unsigned short mins = 0;
  for (unsigned i = 0; i < sizeof(errs)/sizeof(*errs); ++i)
  {
    mins |= (errs[i] == eMin) << i;
  }
  unsigned idx = fmco(mins);

  printf("Error data for HSR @ %p, chan %d:\n", this, chan);
  for (unsigned i = 0; i < sizeof(errs)/sizeof(*errs); ++i)
  {
    printf("  %06x", errs[i]);
    if (i % 8 == 7)  printf("\n");
  }

  smbusEnable(true);
  if (chB) 
    _chB[chan]._eqCtl = eq[idx];
  if (chA)
    _chA[chan]._eqCtl = eq[idx];
  smbusEnable(false);

  return eMin != 0 ? -1 : idx;
}

static unsigned fmco(unsigned short v) // Find Most Contiguous Ones index
{
  if (v == 0)  return 0;

  static const unsigned width = 16;
  unsigned n;
  unsigned cnt = 0;
  unsigned idx = 0;
  unsigned num = 0;

  do
  {
    n = __builtin_ctz(v);
    cnt += n;
    v >>= n;
    n = __builtin_ctz(~v);
    if (n > num)
    {
      num = n;
      idx = cnt;
    }
    cnt += n;
    v >>= n;
  }
  while (cnt < width);

  return idx + (num >> 1);  // Return the index of the center of the group of ones
}
