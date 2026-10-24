/**
 * @file
 * @brief Si570, register access used to read and program the frequency of an Si570 device.
 */
#ifndef Kcu_Si570_hh
#define Kcu_Si570_hh

#include "psdaq/mmhw/Reg.hh"

namespace Drp {
  /** Si570 device accessed as an array of 256 Pds::Mmhw::Reg registers. The methods decode and set a frequency (logged in MHz) through registers 7 to 12, 135 and 137. */
  class Si570 {
  public:
    /** Does nothing; empty body. */
    Si570();
    /** Does nothing; empty body. */
    ~Si570();
  public:
    /** Write 1 to register 135 and poll every 100 us until its bit 0 reads back clear (no timeout). The header comment says this goes back to factory defaults. */
    void   reset();   // Back to factory defaults
    /** Call reset() and read(), then set bit 4 of register 137 (freeze DCO, per the code comment), write HS_DIV, N1 and RFREQ for setting index (0 or 1, default 1) into registers 7 to 12, clear that bit again, set bit 6 of register 135 and call read() once more. The comments give the target as 1300/7 MHz (185.7 MHz). */
    void   program(int index=1); // Set for 185.7 MHz
    /** Decode HS_DIV, N1 and RFREQ from registers 7 to 12 (logging each register) and return 156.25 * HS_DIV * (N1 + 1) * 2^28 / RFREQ, which the log message labels as MHz. The code comment calls this reading the factory calibration for 156.25 MHz. */
    double read();    // Read factory calibration
  private:
    Pds::Mmhw::Reg _reg[256];
  };
};

#endif
