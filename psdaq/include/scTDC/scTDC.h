/**
  @file
*/

#pragma once

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

#include "scTDC_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/**
  @brief    Gives an error description in a text form
  @param    err_code	Integer error code
  @param   err_msg		Pointer to put a text description.

  @note The err_msg array must not be shorter than the value of the  ERRSTRLEN 
  constant.

 */
SCTDCDLL_PUBLIC void sc_get_err_msg(int err_code, char *err_msg);

/**
 * @brief Check type of event data.
 * @param[in] type Event type for comparison.
 * @param[in] event Event under testing.
 * @param[in] event_len_bytes Length of event in bytes.
 * @return int Result of comparison.
 *
 * When reading events via sc_tdc_pipe_read2(), this function can be used to 
 * test whether event is of a particular special type or not.
 *
 * Returns non-zero value if event is of the type specified in the first 
 * argument, otherwise returns zero.
 *
 * @deprecated prefer the USER_CALLBACKS pipe type 
 * @see sc_pipe_open2() with USER_CALLBACKS from enum sc_pipe_type_t (defined in 
 * scTDC_types.h) as a replacement to the sc_tdc_pipe_XYZ() functions.
 */
SCTDCDLL_PUBLIC int
sc_tdc_is_event
(const enum sc_event_type_index type,
const void *event,
const unsigned event_len_bytes);



// -------------------------- api vmaj2 -----------------------------------------

/**
 * @brief Initializes the hardware and loads the initial settings reading
 *  it from ini file.
 * @param ini_filename Name of the inifile used for initialization.
 * @return int device descriptor or error code if less then zero.
 *
 * The function must be called before any other operation and associates
 * a non-negative integer number (device descriptor) with the device. Device
 * descriptor must be used to identify hardware for any other operation.
 *
 * Hardware with the same serial number cannot be opened twice. Call
 * sc_tdc_deinit2() to close and deinitialize hardware.
 */
SCTDCDLL_PUBLIC int sc_tdc_init_inifile(const char *ini_filename);

/**
 * @brief Initializes the hardware and loads the initial settings taken
 * from array of sc_ConfigLine structures.
 * @param confline_array Array of sc_ConfigLine structures which contains
 * configuration needed.
 * @return int device descriptor or error code if less then zero.
 *
 * The function must be called before any other operation and associates a
 * non-negative integer number (device descriptor) with the device. Device
 * descriptor must be used to identify hardware for any other operation.
 *
 * Hardware with the same serial number cannot be opened twice. Call
 * sc_tdc_deinit2() to close and deinitialize hardware.
 *
 * Last structure of the array must contain sc_ConfigLine::section equal NULL.
 */
SCTDCDLL_PUBLIC int
sc_tdc_init_with_config_lines(const struct sc_ConfigLine *confline_array);

/**
 * @brief Deinitialize the hardware
 * @param dev_desc Device descriptor of the hardware to be deinitialized.
 * @return int 0 or error code.
 *
 * Call this function to release device and resources allocated to operate.
 * Hardware is deinitialized after the call.
 */
SCTDCDLL_PUBLIC int sc_tdc_deinit2(const int dev_desc);

/**
 * @brief Get data format of the tdc events.
 * @param dev_desc Device descriptor.
 * @param format Pointer on the structure where format should be placed.
 * @return int 0 or error code.
 *
 * Call this function to get event format got from tdc pipe.
 * See sc_tdc_pipe_open2() for details.
 *
 * <b>Hint:</b> Prefer the USER_CALLBACKS pipe for development of new 
 * applications, which replaces usage of this function.
 */
SCTDCDLL_PUBLIC int
sc_tdc_get_format2(const int dev_desc, struct sc_tdc_format *format);

/**
 * @brief Set a callback to get notifications about end of measurements
 * @param owner a private pointer that is replicated in the first argument
 * of the callback
 * @param cb a callback function receiving the private pointer and a reason
 * code.
 * @return 0 on success, else negative error code.
 * @see SC_TDC_INFO_MEAS_COMPLETE, SC_TDC_INFO_USER_INTERRUPT, 
 * SC_TDC_INFO_BUFFER_FULL, SC_TDC_INFO_HW_IDLE (defined in scTDC_error_codes.h)
 * that the library passes as the second argument into the cb function provided
 * by the user.
 */
SCTDCDLL_PUBLIC int sc_tdc_set_complete_callback2
  (const int, void *owner, void (*cb)(void *, int));

/**
 * @brief Start a measurement.
 * @param dev_desc Device descriptor.
 * @param ms Exposure time in milliseconds.
 * @return 0 on success, else negative error code.
 *
 */
SCTDCDLL_PUBLIC int sc_tdc_start_measure2(const int dev_desc, const int ms);

/**
 * @brief Interrupt a measurement before it completes.
 * @param dev_desc Device descriptor.
 * @return 0 on success, else negative error code.
 *
 */
SCTDCDLL_PUBLIC int sc_tdc_interrupt2(const int dev_desc);

/**
 * @brief Get the size of time bins in nanoseconds.
 * @param dev_desc Device descriptor.
 * @param binsize_ns Pointer where binsize should be stored.
 * @return 0 on success, else negative error code.
 */
SCTDCDLL_PUBLIC int sc_tdc_get_binsize2(const int dev_desc, double *binsize_ns);

/**
 * @brief Query whether the device (hardware + PC-side processing) is idle or in
   a measurement.
 * @param[in] dev_desc Device descriptor.
 * @param[out] status the status value.
 * @return 0 on success, else negative error code.
 *
 * A status value 1 means the device is idle, 0 means that the device is in a
 * measurement.
 */
SCTDCDLL_PUBLIC int sc_tdc_get_status2(const int dev_desc, int *status);

SCTDCDLL_PUBLIC int sc_twi_write2
(const int,
const unsigned char address,
const unsigned char *data,
const size_t size,
const int stop);

SCTDCDLL_PUBLIC int sc_twi_read2
(const int,
const unsigned char address,
unsigned char *data,
const size_t size,
const int stop);

SCTDCDLL_PUBLIC int sc_twi_set_epot2
(const int, unsigned int epot, unsigned int value_number, unsigned char value);

/**
 * @brief Open data pipe.
 * @param dev_desc Device descriptor
 * @param type Pipe type.
 * @param params Pipe parameters.
 * @return A non-negative (>= 0) pipe id in case of success; negative error code
 * in case of failure.
 * The pipe id is to be used in functions sc_pipe_read2() and sc_pipe_close2().
 *
 * The user can open as many pipes as required, even if they are of the same 
 * type.
 * @note In case of internal memory allocation, if data is not read from
 * the pipe after every exposure, the memory consumption of the application
 * will grow. Please make sure to read data as often as necessary.
 * @see enum sc_pipe_type_t
 * @see struct sc_pipe_dld_image_xy_params_t
 * @see struct sc_pipe_dld_image_xt_params_t
 * @see struct sc_pipe_dld_image_yt_params_t
 * @see struct sc_pipe_dld_image_3d_params_t
 * @see struct sc_pipe_dld_sum_histo_params_t
 * @see struct sc_pipe_tdc_histo_params_t
 * @see struct sc_pipe_statistics_params_t
 * @see struct sc_pipe_callback_params_t
 * @see struct sc_pipe_buf_callbacks_params_t
 */
SCTDCDLL_PUBLIC int sc_pipe_open2
(const int dev_desc, const enum sc_pipe_type_t type, const void *params);

/**
 * @brief Close data pipe.
 * @param dev_desc Device descriptor.
 * @param pipe_id Pipe id.
 * @return 0 or error code.
 */
SCTDCDLL_PUBLIC int sc_pipe_close2(const int dev_desc, const int pipe_id);

/**
 * @brief Read data from pipe.
 * @param dev_desc Device descriptor.
 * @param pipe_id Pipe id.
 * @param buffer Pointe to pointer where data block must be stored.
 * @param timeout Timeout in millisecond.
 *
 * This function allocates memory with allocator callback from pipe
 * parameters structure if callback function was installed or allocates
 * memory internally. Then copies data from the last exposure and returns the
 * memory block in buffer. If memory was allocated internally it will be
 * deallocated when next call to the function will be performed or pipe will
 * be closed. If memory allocation callback function was installed in the
 * pipe parameters no any deallocation of the memory will be performed. User
 * must manage memory by him-/her- self.
 *
 * Function returns when: data is available or timeout or pipe is closed.
 */
SCTDCDLL_PUBLIC int sc_pipe_read2
(const int dev_desc, const int pipe_id, void **buffer, unsigned int timeout);

SCTDCDLL_PUBLIC int
sc_tdc_set_flim_scanner2
(const int dd,
unsigned short pixel_interval,
unsigned short pixel_count,
unsigned short line_count,
unsigned int line_delay_interval,
unsigned int multiline_count,
double *corr_table);

SCTDCDLL_PUBLIC int
sc_tdc_set_flim_scanner2ex
(const int dd,
unsigned short pixel_interval,
unsigned short pixel_count,
unsigned short line_count,
unsigned int line_delay_interval,
unsigned int multiline_count,
double *corr_table,
unsigned int flags);

/**
 * @brief see sc_tdc_is_event()
 * @deprecated prefer the USER_CALLBACKS pipe type 
 * @see sc_pipe_open2() with USER_CALLBACKS from enum sc_pipe_type_t (defined in 
 * scTDC_types.h) as a replacement to the sc_tdc_pipe_XYZ() functions.
 */
SCTDCDLL_PUBLIC int
sc_tdc_is_event2
(const enum sc_event_type_index type,
const void *event,
const unsigned event_len_bytes);

/**
 * @brief set channel corrections (aka "channel shifts") and channel mask
 * @param dd Device descriptor.
 * @param ch_count number of values in corrections array
 * @param corrections array of channel correction values
 * @param ch_mask array of masks with as many elements as GPX count
 * @return 0 if successful or (negative) error code
 */
SCTDCDLL_PUBLIC int
sc_tdc_set_corrections2(const int dd, const int ch_count,
  const int *corrections, const unsigned char *ch_mask);

/**
 * @brief get channel corrections (aka "channel shifts")
 * @param dd Device descriptor.
 * @param ch_count receives the number of elements needed for corrections array
 * @param corrections if nullptr, only ch_count is written to, otherwise must
 * point to an array of ints with at least ch_count elements
 * @return 0 if succesful or (negative) error code
 */
SCTDCDLL_PUBLIC int
sc_tdc_get_corrections2(const int dd, int *ch_count, int *corrections);


SCTDCDLL_PUBLIC int sc_tdc_set_common_shift2(const int, const int);
SCTDCDLL_PUBLIC int sc_tdc_set_modulo2(const int, const unsigned int);
SCTDCDLL_PUBLIC int sc_tdc_get_modulo2(const int, unsigned int *);
/**
 * @brief Open data pipe for tdc events.
 * @param dd Device descriptor.
 * @param internal_pipe_size Size of internal data buffer for events in bytes.
 * @param pipe_warning 90% pipe level callback function.
 * @param pipe_alert 99% pipe level callback function.
 * @return int 0 or error code.
 *
 * <b>Hint:</b> Prefer the USER_CALLBACKS pipe for development of new 
 * applications, which replaces usage of this function.
 */
SCTDCDLL_PUBLIC int sc_tdc_pipe_open2
(const int dd, size_t internal_pipe_size,
const struct sc_PipeCbf *pipe_warning,
const struct sc_PipeCbf *pipe_alert);

/**
 * @brief Close data pipe for tdc events.
 * @param dd Device descriptor.
 * @return int 0 or error code.
 *
 * <b>Hint:</b> Prefer the USER_CALLBACKS pipe for development of new 
 * applications, which replaces usage of this function. 
 */
SCTDCDLL_PUBLIC int sc_tdc_pipe_close2(const int dd);

/**
 * @brief Read tdc events from the pipe.
 * @param dd Device descriptor.
 * @param buffer Space for events.
 * @param buffer_size_bytes Size of the buffer in bytes.
 * @param timeout Timeout in milliseconds.
 * @return ssize_t Amount of bytes were copied to the buffer or error code.
 *
 * @see sc_tdc_get_format2(), sc_tdc_is_event2(), sc_tdc_is_event()
 *
 * <b>Hint:</b> Prefer the USER_CALLBACKS pipe for development of new 
 * applications, which replaces usage of this function.
 */
SCTDCDLL_PUBLIC ssize_t sc_tdc_pipe_read2
(const int dd, void *buffer, size_t buffer_size_bytes, unsigned timeout);

/**
 * @brief Get configuration data.
 * @param dd Device descriptor.
 * @param key Configuration key in form 'section:key'
 * @param buf Space for configuration value as a string.
 * @param buf_size Size of buf in bytes.
 * @param def Default value.
 */
SCTDCDLL_PUBLIC int sc_tdc_config_get_key
(const int dd, const char *key, char *buf, size_t buf_size, const char *def);

/**
 * @brief Get scTDC version.
 */
SCTDCDLL_PUBLIC void sc_tdc_config_get_library_version(unsigned [3]);

//v1.1405

/**
 * @brief Get Device Properties. Only use for params_num <= 3.
 * @param dd Device descriptor.
 * @param params_num Number of properties structure.
 * @param params Properties structure casted to void pointer.
 *
 * params_num 1 coresponds to sc_DeviceProperties1 structure.
 * params_num 2 coresponds to sc_DeviceProperties2 structure.
 */
SCTDCDLL_PUBLIC int sc_tdc_get_device_properties
(const int dd, int params_num, void *params);

// v1.3013.13
/**
 * @brief replacement for sc_tdc_get_device_properties when params_num > 3
 */
SCTDCDLL_PUBLIC int sc_tdc_get_device_properties2
(const int dd, int params_num, void *params);

//v1.1406.1
SCTDCDLL_PUBLIC int sc_dld_set_hardware_binning
(const int dd, const struct sc3du_t *binning);

//v1.15xx

/**
  @brief Set Iteration Number.
  @param dd Device descriptor.
  @param itnum Number of iteration per measure.
  @return int 0 or error code.
*/
SCTDCDLL_PUBLIC int sc_tdc_set_iteration_number2(const int dd, int itnum);

//v1.1711.1

SCTDCDLL_PUBLIC int sc_tdc_zero_master_reset_counter
(const int dd);

// v1.3002.0

SCTDCDLL_PUBLIC int sc_flim_get_counters(const int dd,
sc_flimTriggersCounters *);

// v1.3000.5

/**
  @brief Get a copy of the intensity calibration data as an image buffer
  @param dd Device Descriptor
  @param buf user provided buffer to copy to; can be 0 to get the required size
  @param bufsize size of user provided buffer in bytes
  @param size_required where to write the required size of the buffer in bytes
  @param width where to write the width of the image
  @return int 0 or error code
*/
SCTDCDLL_PUBLIC int sc_tdc_get_intens_cal_f(const int dd,
float *buf, size_t bufsize, size_t *size_required, unsigned *width);

// v1.3000.9

/**
 * @brief Set parameters for pulse train simulation generated in the TDC.
 * @param dd Device Descriptor
 * @param start_count number of start pulses; if this number is zero, do not
 * simulate any pulses, regardless of other parameters
 * @param start_period period of start pulses in multiples of 12.5 ns
 * @param start_delay delay of first start pulse to train pulse
 * in multiples of 12.5 ns
 * @param train_count number of train pulses ("train triggers"); if this
 * number is zero, do not generate train pulses, but do generate sub pulses
 * whenever an external train pulse is detected
 * @param train_period period of train pulses in multiples of 12.5 ns
 * @param start_mask start pulse selection bitmask (a repeated concatenation
 * of this bitmask applies to start pulses with indices larger than 32)
 * @return 0 on success, or negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_set_start_sim_params(
  const int dd,
  const unsigned start_count, const unsigned start_period,
  const unsigned start_delay, const unsigned train_count,
  const unsigned train_period, const unsigned start_mask);

SCTDCDLL_PUBLIC int sc_tdc_get_start_sim_params(
  const int dd,
  unsigned* start_count, unsigned* start_period,
  unsigned* start_delay, unsigned* train_count,
  unsigned* train_period, unsigned* start_mask);

// v1.3017.0
/**
 * @brief Create an in-memory "override registry" to store a user-definable set
 * of configuration parameters with values that can deviate from those in the ini
 * file such that during initialization, the alternative values are used without
 * any modification of the actual ini file on hard disk.
 * @return a non-negative handle to an initially empty "override registry",
 * if creation was possible; else negative error code (SC_TDC_ERR_NOMEM)
 */
SCTDCDLL_PUBLIC int sc_tdc_overrides_create();

/**
 * @brief Delete an "override registry" and release its memory.
 * @param handle the non-negative handle returned from sc_tdc_overrides_create()
 * @return 0 on success; else negative error code (SC_TDC_ERR_NO_RESOURCE)
 */
SCTDCDLL_PUBLIC int sc_tdc_overrides_close(int handle);

/**
 * @brief Add an entry to an "override registry" representing a single
 * configuration parameter whose value shall be modified.
 * The entry consists of (1) a section and (2) a key, which correspond to those
 * appearing in ini files, and (3) a value in string form, which will be used
 * instead of the value in the original configuration.
 * @param handle the non-negative handle returned from sc_tdc_overrides_create()
 * @param section the name of the section without the square brackets ([])
 * @param key the name of the parameter
 * @param value the overriding value in any of the string representations that
 * would also work in ini files. However, do not use quotation marks to enclose
 * actual string-typed parameter values.
 * @return 0 on success; negative error code if the handle is unknown; the
 * function does not check whether section, key, or value are valid.
 */
SCTDCDLL_PUBLIC int sc_tdc_overrides_add_entry(
  int handle,
  const char* section,
  const char* key,
  const char* value);

/**
 * @brief see sc_tdc_init_inifile(). Additionally copies the contents from an
 * "override registry" and stores them internally such that values of the
 * configuration parameters listed in that registry override those from the ini
 * file.
 * The registry object is not modified. Call sc_tdc_overrides_close() on the
 * respective handle after initialization to avoid memory leaks unless the
 * registry is planned for later reuse.
 * While a similar effect can be achieved via sc_tdc_init_with_config_lines(),
 * there are some differences:
 * (1) runtime editing of the ini file in DebugLevel > 0 remains possible
 * (2) no parsing of the ini file is required from the application developer
 * (3) no dynamic memory allocation for data structures is required from the
 *     application developer
 * (4) only those parameters that are to be modified need to be added into the
 *     registry
 * @param ini_filename the name of the ini file or the full path to the ini file
 * @param overrides_handle the handle as returned by sc_tdc_overrides_create().
 * If a negative value is passed, no overrides will be in effect, and the
 * function does NOT report an error because of this. In that case, the function
 * has the same effect as calling sc_tdc_init_inifile().
 * @return non-negative device descriptor if the device was successfully
 * initialized; else negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_init_inifile_override(
  const char *ini_filename,
  int overrides_handle);

/**
 * @brief Set the configuration of the pulse generator output that is available
 * on some TDC models
 * @param dev_desc device descriptor
 * @param period the period of the pulse pattern in FPGA clock cycles,
 *   0 or 1 means off
 * @param length the length of the pulse in FPGA clock cycles
 * @return 0 if success, negative error code in case of failure
 */
SCTDCDLL_PUBLIC int sc_tdc_set_pulsegen_config(
  int dev_desc,
  int period,
  int length);

/**
 * @brief Set the configuration of one of two pulse generator outputs that are
 * available on some TDC models (since v1.3021.0)
 * @param dev_desc device descriptor
 * @param channel must be either 0 or 1. Selects Out1 or Out2
 * @param period the period of the pulse pattern in FPGA clock cycles. 0 or 1
 *   means off
 * @param length the length of the pulse in FPGA clock cycles
 * @param initialPhase the initial phase as a number between 0 and period-1.
 *   This phase is established at the start of the measurement.
 * @return 0 if success, negative error code in case of failure
 */
SCTDCDLL_PUBLIC int sc_tdc_set_pulsegen_config2(
  int dev_desc,
  int channel,
  int period,
  int length,
  int initialPhase);

// v1.3020.0
/**
 * @brief Create a reader instance associated with the ini file at the specified
 * file path which provides (read-only) access to parameter values independent
 * of an initialized device instance.
 * @param ini_file_path device descriptor
 * @return a non-negative handle to the reader instance if creation was
 * possible; else negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_create(
  const char* ini_file_path);

/**
 * @brief Read a parameter value as a string
 * @param handle Handle to a reader instance as returned by
 *   sc_tdc_inireader_create()
 * @param section the name of the section inside the ini file without the []
 *   brackets (case sensitive)
 * @param key the name of the parameter (case sensitive)
 * @param valueBuf a buffer where the parameter value will be written to with
 *   null termination
 * @param bufSize size of valueBuf in bytes
 * @return 0 if successful, else negative error code (SC_TDC_ERR_NO_RESOURCE
 * if handle is unknown; SC_TDC_ERR_NO_ENTRY if section-key-combination does
 * not appear in inifile)
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_get_string(
  int handle,
  const char* section,
  const char* key,
  char* valueBuf,
  size_t bufSize);

/**
 * @brief Read a parameter expected to have one of the boolean type values
 * (NO/YES/no/yes/n/y/N/Y/0/1)
 * @param handle Handle to a reader instance as returned by
 *   sc_tdc_inireader_create()
 * @param section the name of the section inside the ini file without the []
 *   brackets (case sensitive)
 * @param key the name of the parameter (case sensitive)
 * @param boolVal target for writing the result, which will be either 0 or 1
 * @return 0 if successful, else negative error code (SC_TDC_ERR_NO_RESOURCE
 * if handle is unknown; SC_TDC_ERR_NO_ENTRY if section-key-combination does
 * not appear in inifile; SC_TDC_ERR_TYPE if parameter value is not one of the
 * recognized boolean type values)
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_get_bool(
  int handle,
  const char* section,
  const char* key,
  int* boolVal
);

/**
 * @brief Read an integer parameter
 * @param handle Handle to a reader instance as returned by
 *   sc_tdc_inireader_create()
 * @param section the name of the section inside the ini file without the []
 *   brackets (case sensitive)
 * @param key the name of the parameter (case sensitive)
 * @param val target for writing the result
 * @return 0 if successful, else negative error code (SC_TDC_ERR_NO_RESOURCE
 * if handle is unknown; SC_TDC_ERR_NO_ENTRY if section-key-combination does
 * not appear in inifile; SC_TDC_ERR_TYPE if parameter value is not an integer)
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_get_long_long(
  int handle,
  const char* section,
  const char* key,
  long long* val
);

/**
 * @brief Read a floating-point valued parameter
 * @param handle Handle to a reader instance as returned by
 *   sc_tdc_inireader_create()
 * @param section the name of the section inside the ini file without the []
 *   brackets (case sensitive)
 * @param key the name of the parameter (case sensitive)
 * @param val target for writing the result
 * @return 0 if successful, else negative error code (SC_TDC_ERR_NO_RESOURCE
 * if handle is unknown; SC_TDC_ERR_NO_ENTRY if section-key-combination does
 * not appear in inifile; SC_TDC_ERR_TYPE if parameter value is not a
 * floating-point value)
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_get_double(
  int handle,
  const char* section,
  const char* key,
  double* val
);

/**
 * @brief Closes the reader instance, releasing associated memory
 * @param handle Handle to the reader instance as returned by
 *   sc_tdc_inireader_create()
 * @return 0 if successful, else negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_inireader_close(
  int handle);

// v1.3023.0

/**
 * @brief sc_tdc_trigger_master_reset trigger a master reset in the GPX chip
 * @param dd the device descriptor
 * @return 0 if successful, else negative error code
 */
SCTDCDLL_PUBLIC int sc_tdc_trigger_master_reset(
  const int dd
);

#ifdef __cplusplus
}
#endif // __cplusplus

#define ERRSTRLEN				256

// older code bases of end-user software: define SC_TDC_AUTOINCLUDE_CAM_H
// in the build tool
#ifdef SC_TDC_AUTOINCLUDE_CAM_H
#include "scTDC_cam.h"
#endif
