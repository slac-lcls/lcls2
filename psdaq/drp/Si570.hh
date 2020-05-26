#ifndef Kcu_Si570_hh
#define Kcu_Si570_hh

namespace Drp {
  class Si570 {
  public:
    Si570(int fd, unsigned off);
    ~Si570();
  public:
    void   reset();   // Back to factory defaults
    void   program(); // Set for 185.7 MHz
    double read();    // Read factory calibration
  private:
    int      _fd;
    unsigned _off;
  };
};

#endif
