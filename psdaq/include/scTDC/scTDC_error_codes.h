#pragma once

/**
  @file
*/

/**
  General or unknown error.
*/
#define SC_TDC_ERR_DEFAULT                                -1

// (never returned)
#define SC_TDC_ERR_INIFILE                                -2

// (never returned)
#define SC_TDC_ERR_TDCOPEN                                -3
/**
  System has not enough memory to perform operation.
*/
#define SC_TDC_ERR_NOMEM                                  -4

// (never returned)
#define SC_TDC_ERR_SERIAL                                 -5

// (never returned)
#define SC_TDC_ERR_TDCOPEN2                               -6

// (never returned)
#define SC_TDC_ERR_PARAMETER                              -7

// (never returned)
#define SC_TDC_ERR_SMALLBUFFER                            -8
/**
  Inifile cannot be found or has incorrect syntax.
*/
#define SC_TDC_ERR_BADCONFI                               -9
/**
  Device is not initialized or bad device descriptor.
*/
#define SC_TDC_ERR_NOTINIT                                -10
/**
  Device is not ready to operate.
*/
#define SC_TDC_ERR_NOTRDY                                 -11
/**
  Could not load device class library. Make sure that it is present, has
  correct architecture string and dependencies.
*/
#define SC_TDC_ERR_DEVCLS_LD                              -12
/**
  Device class library has unsupported version. Update or use appropriate
  version.
*/
#define SC_TDC_ERR_DEVCLS_VER                             -13
/**
  Could not initialize device class library. Make sure that firmware file is
  present.
*/
#define SC_TDC_ERR_DEVCLS_INIT                            -14
/**
  Could not initialize fpga. Make sure that the firmware file is present and correct.
*/
#define SC_TDC_ERR_FPGA_INIT                              -15
/**
  Subsystem is already initialized. To reinitialize, deinitialize first.
*/
#define SC_TDC_ERR_ALRDYINIT                              -16
/**
  Timeout during reading of data.
*/
#define SC_TDC_ERR_TIMEOUT                                -17
/**
  No simulation data file found. Make sure that it is present and correct.
*/
#define SC_TDC_ERR_NOSIMFILE                              -18
/**
  Spurious wakeup.
*/
#define SC_TDC_ERR_SPURIOUS_WAKEUP                        -19
/**
  Synchronisation error.
*/
#define SC_TDC_ERR_SYNC                                   -20
/**
  Could not reset GPX. Communication with GPX broken.
*/
#define SC_TDC_ERR_GPX_RST                                -21
/**
  Could not lock PLL.
*/
#define SC_TDC_ERR_GPX_PLL_NLOCK                          -22

// (never returned)
#define SC_TDC_ERR_USB_COMM                               -30

// (never returned)
#define SC_TDC_ERR_CORR_SET                                -40
#define SC_TDC_ERR_BIN_SET                                -41
/**
  Setting of ROI failed
*/
#define SC_TDC_ERR_ROI_SET                                -42

// (never returned)
#define SC_TDC_ERR_FMT_SET                                -43

// (never returned)
#define SC_TDC_ERR_FMT_UNSUPPORT                          -44
/**
  Attempt to set an incorrect ROI
*/
#define SC_TDC_ERR_ROI_BAD                                -45
/**
  Attempt to set an incorrect ROI
*/
#define SC_TDC_ERR_ROI_TOOBIG                             -46
/**
  @deprecated returned from sc_tdc_alloc_buffer() and
  sc_tdc_alloc_buffer_v()
*/
#define SC_TDC_ERR_BUFSIZE                                -47

// (never returned)
#define SC_TDC_ERR_GPX_FMT_UNSUPPORT                      -48
/**
  Could not set GPX format
*/
#define SC_TDC_ERR_GPX_FMT_SET                            -49
/**
  @deprecated returned from sc_tdc_alloc_buffer() and
  sc_tdc_alloc_buffer_v()
*/
#define SC_TDC_ERR_FMT_NDEF                               -50
/**
  Function called with bad arguments
*/
#define SC_TDC_ERR_BAD_ARGUMENTS                          -51
/**
  Unable to set fifo reading address
*/
#define SC_TDC_ERR_FIFO_ADDR_SET                          -60
/**
  Unable to set mode
*/
#define SC_TDC_ERR_MODE_SET                               -61
/**
  Could not start gpx reading
*/
#define SC_TDC_ERR_START_FAIL                             -62
/**
  Unable to set timer
*/
#define SC_TDC_ERR_TIMER_SET                              -63
/**
  Unable to set time range extender
*/
#define SC_TDC_ERR_TIMER_EX_SET                           -64
/**
  Unable to set start frequency divider
*/
#define SC_TDC_ERR_STRT_FREQ_DIV_SET                      -65
/**
  Unable to set start frequency period
*/
#define SC_TDC_ERR_STRT_FREQ_PERIOD_SET                   -66

#define SC_TDC_ERR_STRT_FREQ_PERIOD_GET                   -67
#define SC_TDC_ERR_STRT_FREQ_PERIOD_VALUE                 -68
#define SC_TDC_ERR_MODULO_VALUE_SET                       -69

/**
  TWI module is not available
*/
#define SC_TDC_ERR_TWI_NO_MODULE                          -70
/**
  Unable to operate TWI module
*/
#define SC_TDC_ERR_TWI_FAIL                               -71
/**
  No acknowledge received from TWI slave
*/
#define SC_TDC_ERR_TWI_NACK                               -72
/**
  Digital potentiometer is not available
*/
#define SC_TDC_ERR_POT_NO                                 -73
/**
  Cannot set digital potentiometer
*/
#define SC_TDC_ERR_POT_SET                                -74
/**
  Cannot set FLIM scanner parameters
*/
#define SC_TDC_ERR_FLIM_PARM_SET                          -80
/**
  Cannot open line_cor.txt file
*/
#define SC_TDC_ERR_OPEN_LINE_CORR_FILE                    -81
/**
  Wrong line_cor.txt file
*/
#define SC_TDC_ERR_WRONG_LINE_CORR_FILE                   -82
/**
  No connection to the device
*/
#define SC_TDC_ERR_CONNLOST                               -90
/**
  Unable to set/get CMOS register
*/
#define SC_TDC_ERR_CMOS_REG                               -95
/**
  Could not load library
*/
#define SC_TDC_ERR_NO_LIBRARY                             -96
/**
  Could not find library symbol
*/
#define SC_TDC_ERR_NO_LIBRARY_SYM                         -97
/**
  No device found
*/
#define SC_TDC_ERR_NO_DEVICE                              -98
/**
  A non-existent resource handle was specified
*/
#define SC_TDC_ERR_NO_RESOURCE                            -99
/**
  No intensity calibration available
*/
#define SC_TDC_ERR_NO_INTENS_CAL                          -110
/**
  Unable to retrieve period of tag pulses
*/
#define SC_TDC_ERR_TAG_FREQ_PERIOD_GET                    -120
/**
  Entry with specified key not found
*/
#define SC_TDC_ERR_NO_ENTRY                               -130
/**
  Type is not applicable
*/
#define SC_TDC_ERR_TYPE                                   -131
/**
  No enough resource available
*/
#define SC_TDC_ERR_SYSTEM                                 -1000
/**
  Insufficient permissions of the current user
*/
#define SC_TDC_ERR_PERMISSION                             -1001
/**
  Problem while working with file system functions
*/
#define SC_TDC_ERR_FILESYSTEM                             -1002
/**
  Feature not implemented
*/
#define SC_TDC_ERR_NOT_IMPL                               -9000
/**
  Feature not supported by hardware
*/
#define SC_TDC_ERR_NO_HARDWARE_SUPPORT                       -9001

//used only for callback
/**
  Reason code used in callback registered via sc_tdc_set_complete_callback2().
  Indicates regular completion of the measurement.
*/
#define SC_TDC_INFO_MEAS_COMPLETE                         1
/**
  Reason code used in callback registered via sc_tdc_set_complete_callback2().
  Indicates measurement interrupted by user.
*/
#define SC_TDC_INFO_USER_INTERRUPT                        2
/**
  Reason code used in callback registered via sc_tdc_set_complete_callback2().
  Indicates aborted measurement because a buffer capacity was exhausted.
*/
#define SC_TDC_INFO_BUFFER_FULL                           3
/**
  Reason code used in callback registered via sc_tdc_set_complete_callback2().
  Indicates that the hardware reached an idle state after a measurement while
  data is still being processed on the PC side. Typically followed by a
  callback with reason SC_TDC_INFO_MEAS_COMPLETE.
*/
#define SC_TDC_INFO_HW_IDLE                               4

