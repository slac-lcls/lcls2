#ifndef Cphw_HsRepeater_hh
#define Cphw_HsRepeater_hh

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/Reg64.hh"
#include <stdio.h>

namespace Pds {
  namespace Cphw {

    class HsChannel {
    public:
      void    dump() const;
      void    save(FILE*) const;
      void    load(FILE*);
    public:
      bool    sigDetPreset() const;
      void    sigDetPreset(bool);
      bool    sigDetReset() const;
      void    sigDetReset(bool);
    public:
      uint8_t rxdet() const;
      void    rxdet(uint8_t);
      bool    idle_sel() const;
      void    idle_sel(bool);
      bool    idle_auto() const;
      void    idle_auto(bool);
    public:
      void    eqCtl(unsigned);
    public:
      uint8_t vodCtl() const;
      void    vodCtl(uint8_t);
      bool    mode_sel() const;
      void    mode_sel(bool);
      bool    shortCircProt() const;
      void    shortCircProt(bool);
    public:
      uint8_t demCtl() const;
      void    demCtl(uint8_t);
      uint8_t mode_detStatus() const;
      void    mode_detStatus(uint8_t);
      bool    rxdetStatus() const;
      void    rxdetStatus(bool);
    public:
      uint8_t idleThd() const;
      void    idleThd(uint8_t);
      uint8_t idleTha() const;
      void    idleTha(uint8_t);
    private:
      uint32_t    _reserved;            // Too bad this can't be at the end
    public:
      //  0x00 - RW: Signal detector force to off or on
      //  [1]    on   SD Preset
      //  [2]    off  SD Reset
      Cphw::Reg   _sigDetFrc;
      //  0x04 - RW: Receive detect
      //  [3:2]  RXDET
      //  [4]    IDLE_SEL
      //  [5]    IDLE_AUTO
      Cphw::Reg   _rxDet;
      //  0x08 - RW: Equalization control
      //  [7:0]  EQ Control
      Cphw::Reg   _eqCtl;
      //  0x0C - RW: VOD
      //  [2:0]  VOD control
      //  [6]    MODE_SEL
      //  [7]    Short Circuit Protection
      Cphw::Reg   _vod;
      //  0x10 - RW: DEM
      //  [2:0]  DEM Control
      //  [6:5]  MODE_DET STATUS
      //  [7]    RXDET STATUS
      Cphw::Reg   _dem;
      //  0x14 - RW: Idle threshold
      //  [1:0]  IDLE thd  Deassert threshold
      //  [3:2]  IDLE tha  Assert threshold
      Cphw::Reg   _idleThrsh;
    };

    class HsRepeater {
    public:
      enum { NChannels=4 };
      typedef unsigned (*MeasFn)(void*);
    public:
      HsRepeater();
    public:
      void     init();  // Initialize with defaults
      void     dump(unsigned links = (1 << NChannels)-1) const;
      void     save(FILE*) const;
      void     load(FILE*);
    public:
      unsigned scanLink(unsigned chan, bool chA, bool chB, MeasFn measFn, void* arg);
    public:
      bool    eepromReadDone() const;
      uint8_t addressBits()    const;
    public:
      bool    pwdnChan(uint8_t) const;
      void    pwdnChan(uint8_t, bool);
    public:
      bool    pwdnOverride() const;
      void    pwdnOverride(bool);
      uint8_t lpbkCtl() const;
      void    lpbkCtl(uint8_t);
    public:
      bool    smbusEnable() const;
      void    smbusEnable(bool);
    public:
      void    resetSmbus();
      void    resetRegs();
    public:
      bool    modePin() const;
      void    modePin(bool);
      bool    rxdetPin() const;
      void    rxdetPin(bool);
      bool    idleCtl() const;
      void    idleCtl(bool);
      bool    sd_thPin() const;
      void    sd_thPin(bool);
    public:
      bool    sigDet(uint8_t) const;
    public:
      uint8_t reducedGain() const;
      void    reducedGain(uint8_t);
      uint8_t fastIdle() const;
      void    fastIdle(uint8_t);
      uint8_t highIdle() const;
      void    highIdle(uint8_t);
    public:
      uint8_t devVer() const;
      uint8_t devId()  const;
    public:
      //  0x0000 - RO: Device address readback
      //  [2]      EEPROM Read Done
      //  [6:3]    Address bit AD[3:0]
      Cphw::Reg   _devAddr;
      //  0x0004 - RW: Power down per channel
      //  [7:0]    PWDN CHx
      Cphw::Reg   _pwdnChans;
      // Revisit: Shouldn't the following be RW?
      //  0x0008 - RO: Override power down
      //  [0]      Override PWDN pin
      //  [5:4]    LPBK Control
      Cphw::Reg   _ovrPwdn;
    private:
      uint32_t    _reserved_12[3];
    public:
      //  0x0018 - RW: Enable slave register write
      //  [3]      SMBus Register Enable
      Cphw::Reg   _slvRegCtl;
      //  0x001c - RW: Digital Reset and Control
      //  [5]      Reset SMBus Master
      //  [6]      Reset Registers
      Cphw::Reg   _digRstCtl;          // Revisit: Not impled?
      //  0x0020 - RW: Override Pin Control
      //  [2]      Override MODE
      //  [3]      Override RXDET
      //  [4]      Override IDLE
      //  [6]      Override SD_TH
      Cphw::Reg   _ovrPinCtl;          // Revisit: Not impled?
    private:
      uint32_t    _reserved_36;
    public:
      //  0x0028 - RO: Signal detect monitor
      //  [7:0]    SD_TH Status
      Cphw::Reg   _sigDetMon;
    private:
      uint32_t    _reserved_44;
    public:
      //  -4 here because there's a reserved word the front of the struct:
      //  0x0034-4, 0x0050-4, 0x006c-4, 0x0088-4: CH0 - CHB0 to CH3 - CHB3
      HsChannel   _chB[NChannels];
      //  0x00A0 - RW: Signal detect control
      //  [1:0]    Reduced SD Gain
      //  [3:2]    Fast IDLE
      //  [5:4]    High IDLE
      Cphw::Reg   _sigDetCtl;
      //  -4 here because there's a reserved word the front of the struct:
      //  0x00A8-4, 0x00C4-4, 0x00E0-4, 0x00FC-4: CH0 - CHA0 to CH3 - CHA3
      HsChannel   _chA[NChannels];
    private:
      uint32_t    _reserved_276[12];
    public:
      // 0x0144 - RO: Device ID
      Cphw::Reg   _devID;
    private:
      // Round up to allow making an array of HsRepeaters each of size 0x10000 bytes
      uint32_t    _reserved_284[(0x10000 - 0x0148)/sizeof(uint32_t)];
    };
  };
};

#endif
