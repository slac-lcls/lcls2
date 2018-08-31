#ifndef XtcData_XtcMonitorMsg_hh
#define XtcData_XtcMonitorMsg_hh

#include <stdint.h>

namespace XtcData {
  class XtcMonitorMsg {
    enum { SizeMask   = 0x0fffffff };
    enum { SerialShift = 28 };
  public:
    XtcMonitorMsg() : _bufferIndex(0),
                      _numberOfBuffers(0),
                      _sizeOfBuffers(0),
                      _reserved(0) {}
    XtcMonitorMsg(int bufferIndex) : _bufferIndex(bufferIndex),
                                     _numberOfBuffers(0),
                                     _sizeOfBuffers(0),
                                     _reserved(0) {}
    ~XtcMonitorMsg() {};
  public:
    int bufferIndex     () const { return _bufferIndex; }
    int numberOfBuffers () const { return _numberOfBuffers&0xff; }
    int numberOfQueues  () const { return (_numberOfBuffers>>8)&0xff; }
    int sizeOfBuffers   () const { return _sizeOfBuffers&SizeMask; }
    bool serial         () const { return return_queue()==0; }
    int return_queue    () const { return (_numberOfBuffers>>16)&0xff; }
  public:
    XtcMonitorMsg* bufferIndex(int b) {_bufferIndex=b; return this;}
    void numberOfBuffers      (int n) {_numberOfBuffers &= ~0xff; _numberOfBuffers |= ((n&0xff)<<0); }
    void numberOfQueues       (int n) {_numberOfBuffers &= ~0xff00; _numberOfBuffers |= ((n&0xff)<<8); }
    void sizeOfBuffers        (int s) {_sizeOfBuffers = (_sizeOfBuffers&~SizeMask) | (s&SizeMask);}
    void return_queue         (int q) {_numberOfBuffers &= ~0xff0000; _numberOfBuffers |= ((q&0xff)<<16); }
  public:
    static void sharedMemoryName     (const char* tag, char* buffer);
    static void eventInputQueue      (const char* tag, unsigned client, char* buffer);
    static void eventOutputQueue     (const char* tag, unsigned client, char* buffer);
    static void transitionInputQueue (const char* tag, unsigned client, char* buffer);
    static void discoveryQueue       (const char* tag, char* buffer);
    static void registerQueue        (const char* tag, char* buffer, int id);
  private:
    int32_t  _bufferIndex;
    int32_t  _numberOfBuffers;
    uint32_t _sizeOfBuffers;
    uint32_t _reserved;
  };
};

#endif
