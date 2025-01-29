#pragma once

/**
  @file
*/

enum SC_Cmos_ConvGain {
  SC_CMOS_CONVGAIN_LOW = 0,
  SC_CMOS_CONVGAIN_HIGH = 1
};

/**
  cmos feature support : shutter mode
*/
#define SC_CMOS_FEATURE_SHUTTER_MODE             0x1

/**
  cmos feature support : exposure dual slope
*/
#define SC_CMOS_FEATURE_EXPOSURE_DUAL_SLOPE      0x2

/**
  cmos feature support : exposure triple slope
*/
#define SC_CMOS_FEATURE_EXPOSURE_TRIPLE_SLOPE    0x4

/**
  cmos feature support : frame count
*/
#define SC_CMOS_FEATURE_FRAME_COUNT              0x8

/**
  cmos feature support : analog gain
*/
#define SC_CMOS_FEATURE_ANALOG_GAIN             0x10

/**
  cmos feature support : digital gain
*/
#define SC_CMOS_FEATURE_DIGITAL_GAIN            0x20

/**
  cmos feature support : black offset
*/
#define SC_CMOS_FEATURE_BLACK_OFFSET            0x40

/**
  cmos feature support : black cal offset
*/
#define SC_CMOS_FEATURE_BLACK_CAL_OFFSET        0x80

/**
  cmos feature support : smoother 1
*/
#define SC_CMOS_FEATURE_SMOOTHER_ONE           0x100

/**
  cmos feature support : smoother 2
*/
#define SC_CMOS_FEATURE_SMOOTHER_TWO           0x200

/**
  cmos feature support : white pixel min
*/
#define SC_CMOS_FEATURE_WHITE_PIXEL_MIN        0x400

/**
  cmos feature support : set region of interest
*/
#define SC_CMOS_FEATURE_ROI                    0x800

/**
  cmos feature support : set bit mode
*/
#define SC_CMOS_FEATURE_BITMODE               0x1000

/**
  cmos feature support : fpga-side accumulation of images
*/
#define SC_CMOS_FEATURE_IMG_ACCU              0x2000

/**
  cmos feature support : bit transfer mode (transfer_bytes = compatibility alias)
*/
#define SC_CMOS_FEATURE_TRANSFER_BYTES        0x4000
#define SC_CMOS_FEATURE_BIT_TRANSFER_MODE     0x4000

/**
  cmos feature support : bit shift
*/
#define SC_CMOS_FEATURE_BITSHIFT              0x8000

/**
  cmos feature support : conversion gain
*/
#define SC_CMOS_FEATURE_CONVGAIN             0x10000
