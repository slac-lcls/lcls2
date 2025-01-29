/**
  @file
*/

/** @file */

#pragma once

#ifndef SCTDCDLL_PUBLIC
#if defined _WIN32 || defined __CYGWIN__
  #ifdef SCTDC_EXPORTS
    #define SCTDCDLL_PUBLIC __declspec(dllexport)
    #define WIN32_LEAN_AND_MEAN
  #else
    #define SCTDCDLL_PUBLIC __declspec(dllimport)
  #endif
#elif __linux__
  #ifdef SCTDC_EXPORTS
    #define SCTDCDLL_PUBLIC __attribute__ ((visibility("default")))
  #else
    #define SCTDCDLL_PUBLIC
  #endif
  #include <sys/types.h>
#else
  #error platform is not supported
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
  @brief Set cmos and smoother parameters.
  @param dd Device descriptor.
  @param params Cmos and smoother parameters structure.
  @return int 0 or error code.
*/
SCTDCDLL_PUBLIC int sc_tdc_set_cmos_and_smoothers_params
(const int dd, const struct sc_CmosSmootherParameters *params);

/**
  @brief Set blob parameters.
  @param dd Device descriptor.
  @param params Blob parameters.
  @return int 0 or error code.
*/
SCTDCDLL_PUBLIC int sc_tdc_set_blob_parameters
(const int dd, const struct sc_BlobParameters *params);

/**
  @brief Get cmos and smoother parameters.
  @param dd Device descriptor.
  @param params Cmos and smoother parameters structure.
  @return int 0 or error code.

  Function reads default configuration for cmos and smoother
  from the inifile. Setting parameters with
  sc_tdc_set_cmos_and_smoothers_params()
  does not change default parameters.
*/
SCTDCDLL_PUBLIC int sc_tdc_get_cmos_and_smoothers_params
(const int dd, struct sc_CmosSmootherParameters *params);

/**
  @brief Get blob parameters.
  @param dd Device descriptor.
  @param params Blob parameters structure.
  @return int 0 or error code.

  Function read default configuration for blob algorithm
  from inifile. Setting parameters with
  sc_tdc_set_blob_parameters does not change
  default parameters.
*/
SCTDCDLL_PUBLIC int sc_tdc_get_blob_parameters
(const int dd, struct sc_BlobParameters *params);

//v1.3001.0

/**
  @brief Set blob algorithm.
  @param dd Device descriptor.
  @param blob Library name where blob algorithm implemented.
  @return int 0 or error code.
*/
SCTDCDLL_PUBLIC int sc_tdc_set_blob
(const int dd, const char *blob);

/**
  @brief Get default blob algorithm.
  @param dd Device descriptor.
  @param str Buffer for blob name string.
  @param str_len Buffer for blob name string length.
  @param str_len_ret Used memory space for blob name string.
  @return int 0 or error code.
*/
SCTDCDLL_PUBLIC int sc_tdc_get_blob
(const int dd, char *str, size_t str_len, size_t *str_len_ret);

/**
 * @brief Set a region of interest in a camera. Restriction to a subset of the
 * sensor area may enable faster frame rates. To go back to the full sensor area,
 * use x_min = 0, x_max = 8192, y_min = 0, y_max = 8192 (or any values for
 * x_max, y_max that are larger than the native number of pixels on the sensor).
 * @param dd Device Descriptor
 * @param x_min left border
 * @param x_max right border
 * @param y_min top border
 * @param y_max bottom border
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_set_roi(const int dd, const unsigned x_min,
  const unsigned x_max, const unsigned y_min, const unsigned y_max);

/**
 * @brief Get the currently set region of interest for a camera
 * @param dd Device Descriptor
 * @param x_min left border
 * @param x_max right border
 * @param y_min top border
 * @param y_max bottom border
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_roi(const int dd, unsigned* x_min,
  unsigned* x_max, unsigned* y_min, unsigned* y_max);

/**
 * @brief Query the supported features of a camera
 * @param dd Device Descriptor
 * @param f filled with a bitmask that represents the supported features
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_supported_features(const int dd, unsigned* f);

/**
 * @brief Set exposure and/or frames.
 * @param dd Device Descriptor
 * @param e exposure in microseconds; specify zero to leave exposure unchanged
 * @param f number of frames; specify zero to leave the number of frames unchanged
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_set_exposure(const int dd, unsigned e, unsigned f);

/**
 * @brief Get width and height of the maximum possible region of interest,
 * corresponding to the sensor size in pixels
 * @param dd Device Descriptor
 * @param width of the maximum possible region of interest
 * @param height of the maximum possible region of interest
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_maxsize(const int dd, unsigned* width,
  unsigned* height);

/**
 * @brief Get the temperatures in degree Celsius of two sensors inside the
 * housing
 * @param dd Device Descriptor
 * @param fpga temperature 1 (FPGA board)
 * @param cmos temperature 2 (CMOS board)
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_temperatures(const int dd, double* fpga,
  double* cmos);

/**
 * @brief Set the fan speed
 * @param dd Device Descriptor
 * @param fanspeed accepted values range from 0 to 255 (the threshold for the
 * fan to start rotating varies and may be somewhere between 1 and 100)
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_set_fanspeed(const int dd, int fanspeed);

/**
 * @brief get properties
 * @param dd Device Descriptor
 * @param ptype properties type
 * @param dest pointer to the cam properties data structure corresponding to
 * the value of ptype (one of sc_CamProperties...)
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_properties(
  const int dd, const int ptype, void* dest);

/**
 * @brief set one of the model-specific, more specialized parameters
 * @param dd Device Descriptor
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_set_parameter(
  const int dd, const char* name, const char* value);

/**
 * @brief get the value of one of the model-specific, more specialized parameters
 * @param dd Device Descriptor
 * @param name parameter name
 * @param value pointer to a buffer where to write the value, or nullptr
 * @param str_len if value == nullptr, the required buffer size is returned in
 * (*str_len); if value != nullptr, (*str_len) is interpreted as the size of the
 * buffer provided at (*value)
 * @return
 */
SCTDCDLL_PUBLIC int sc_tdc_cam_get_parameter(
  const int dd, const char* name, char* value, size_t* str_len);

#ifdef __cplusplus
}
#endif // __cplusplus
