#pragma once

/** @file */

struct sc_CamProperties1
{
  int supports_blob; /**< if > 0, the camera supports blob recognition */
  int supports_upscaling; /**< if > 0, the camera supports blob coordinates with
                               resolution beyond the pixel grid of the sensor */
  int supports_adc; /**< if > 0, the camera supports an ADC input */
};

struct sc_CamProperties2
{
  int supports_convolution_mask;
};

struct sc_CamProperties3
{
  int sensor_max_intensity; /**< the maximum intensity level on the sensor
    according to the sensor type and configuration (camera frames read by the
    application may deliver down-scaled intensities compared to what is captured
    by the sensor due to the BitShift parameter in the [CMOS] section of the ini
    file */
  int frame_max_intensity; /**< the maximum intensity level delivered in
    raw image camera frames */
};

enum sc_cam_pixelformat_t
{
  SC_CAM_PIXELFORMAT_UINT8 = 0, /**< pixel values are unsigned 8-bit integers */
  SC_CAM_PIXELFORMAT_UINT16 = 1 /**< pixel values are unsigned 16-bit integers */
};

enum sc_cam_frame_meta_flags_t {
  SC_CAM_FRAME_HAS_IMAGE_DATA = 1, /**< if set, the #sc_cam_frame_meta_t block
                                        is followed by pixel data */
  SC_CAM_FRAME_IS_LAST_FRAME = 2   /**< if set, this frame is the last frame
                                        of the measurement */
};

struct sc_cam_frame_meta_t
{
  unsigned data_offset; /**< memory address offset to the image data */
  unsigned frame_idx;   /**< index of frame within measurement */
  unsigned long long frame_time; /**< time stamp of the frame */
  unsigned short width; /**< width of the image / currently set ROI */
  unsigned short height; /**< height of the image / currently set ROI */
  unsigned short roi_offset_x; /**< horizontal position of the ROI on sensor */
  unsigned short roi_offset_y; /**< vertical position of the ROI on sensor */
  unsigned short adc; /**< ADC value, digitized voltage on ADC hardware input */
  unsigned char pixelformat;  /**< see #sc_cam_pixelformat_t */
  unsigned char flags;     /**< see #sc_cam_frame_meta_flags_t */
  unsigned char bitdepth;  /**< the maximum number of bits in use */
  unsigned char hdr_subframe_count;
  unsigned char hdr_exposure_step;
  unsigned char internal_1_;
  unsigned hdr_subframe_interval;
  unsigned char reserved[4]; /**< no data, inserted for memory alignment */
};

struct sc_cam_blob_meta_t
{
  unsigned data_offset; /**< memory address offset to the blob data */
  unsigned nr_blobs;    /**< number of blobs */
};

struct sc_cam_blob_position_t
{
  float x;
  float y;
};
