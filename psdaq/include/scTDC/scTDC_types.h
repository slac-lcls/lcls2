/**
 @file
*/

#pragma once

#ifndef ssize_t
#ifdef __linux__
  #include <sys/types.h>
#elif _WIN32
  #ifdef _MSC_VER
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
  #elif __MINGW32__
    #include <sys/types.h>
  #endif
#endif
#endif

/**
 * @brief Used in sc_tdc_format::flow_control_flags
 * @details Start and Beginning of statistics is placed in the tdc event stream
 * if this flag is on.
 */
#define SEPARATORS_TO_FLOW 0x01

/**
 * @brief Used in sc_tdc_format::flow_control_flags
 * @details 1024 bytes of raw statistics is placed in the endo of tdc event stream
 * if this flag is on.
 */
#define STATISTICS_TO_FLOW 0x02

/**
 * @brief Used in sc_tdc_format::flow_control_flags
 * @details Millisecond signs are placed in the tdc event stream
 * if this flag is on.
 */
#define MILLISECONDS_TO_FLOW 0x04

/**
 * @brief Used in sc_tdc_set_flim_scanner2ex()
 * @details Both way scanning mode is active
 * if this flag is on.
 */

#define FLIM_BOTH_WAY_SCAN 0x01

/**
 * @brief Used in sc_tdc_set_flim_scanner2ex()
 * @details Image data are swapped
 * if this flag is on.
 */
#define FLIM_XY_SWAP 0x02


/**
 * @struct sc3d_t
 * @brief Signed 3d point.
 */
struct sc3d_t
{
  int x; /**< x coordinate. */
  int y; /**< y coordinate. */
  long long time; /**< time coordinate. */
};

/**
 * @struct sc3du_t
 * @brief Unsigned 3d point
 */
struct sc3du_t
{
  unsigned int x; /**< x coordinate. */
  unsigned int y; /**< y coordinate. */
  unsigned long long time; /**< time coordinate. */
};

/**
 * @struct roi_t
 * @brief Region of interest in a three-dimensional coordinate system spanned
 * by detector position (x, y) and the time coordinate. Offsets mark the 
 * margins at the lower coordinate ends, sizes define the extension towards
 * the higher coordinate end.
 */
struct roi_t
{
  struct sc3d_t offset; /**< Roi offset */
  struct sc3du_t size; /**< Roi size */
};

/**
 * @struct statistics_t
 * @brief Measurement statistics. The first array index corresponds to the 
   subdevice number, the second array index corresponds to channels. If there
   are no subdevices, data is found only for the first array index being set to
   0.
 */
struct statistics_t
{
  /** Tdc data per channel recognized by fpga. */
  unsigned int counts_read[4][16];
  /** Tdc data per channel transferred from fpga to cpu. */
  unsigned int counts_received[4][16];
  /** Dld events recognized in fpga. */
  unsigned int events_found[4];
  /** Dld events recognized in fpga which fits in hardware roi. */
  unsigned int events_in_roi[4];
  /** Dld events transferred from fpga to cpu. */
  unsigned int events_received[4];
  unsigned int counters[4][16];
  unsigned int reserved[52];
};

struct sc_tdc_channel_statistics_t
{
  unsigned long long counts_read;
  unsigned long long counts_received;
  unsigned long long counter;
  unsigned long long reserved[5];
};

struct sc_dld_device_statistics_t
{
  unsigned long long events_found;
  unsigned long long events_in_roi;
  unsigned long long events_received;
  unsigned long long reserved[5];
};

/**
 * @brief Size and data type of pixel values (in images), or of histogram 
 * elements (in time histograms).
 */
enum bitsize_t
{
  BS8 = 0, /**< Unsigned 8-bit (1 byte) integer. */
  BS16 = 1, /**< Unsigned 16-bit (2 bytes) integer. */
  BS32 = 2, /**< Unsigned 32-bit (4 bytes) integer. */
  BS64 = 3, /**< Unsigned 64-bit (8 bytes) integer. */
  F32 = 4, /**< Single-precision IEEE 754 floating-point (4 bytes). */
  F64 = 5 /**< Double-precision IEEE 754 floating-point (8 bytes). */
};

/**
 * @struct sc_tdc_format
 * @brief Contains sizes and offsets of data bitfields.
 * @details Zero value of the field means that field is not present in the event.
 */
struct sc_tdc_format
{
  unsigned char total_bits_length; /**< Length of one event in bits.
                                  Currently can be only 8, 16, 32 and 64*/
  unsigned char channel_offset; /**< Offset of channel field. Mostly used
                                    in tdc mode */
  unsigned char channel_length; /**< Length of channel field. Mostly used
                                    in tdc mode. Channel field contains information
                                    in which channel of TDC event occured. */
  unsigned char time_data_offset; /**< Offset of time data data field. Mostly used
                                    in tdc mode */
  unsigned char time_data_length; /**< Length of time_data field. Mostly used
                                    in tdc mode. time_data field contains
                                    information about time when event occurs [binsize] */
  unsigned char time_tag_offset;
  unsigned char time_tag_length;
  unsigned char start_counter_offset; /**< Offset of start_counter data field */
  unsigned char start_counter_length; /**< Length of start_counter data field.
                                    start_counter data field contains information
                                    about start counter value. See documentation
                                    to the device for more info about start counter
                                    value */
  unsigned char dif1_offset; /**< Offset of x coordinate of the event */
  unsigned char dif1_length; /**< Length of x coordinate of the event. Mostly
                            used in dld mode.*/
  unsigned char dif2_offset; /**< Offset of y coordinate of the event */
  unsigned char dif2_length; /**< Length of y coordinate of the event. Mostly
                            used in dld mode.*/
  unsigned char sum_offset; /**< Offset of time coordinate data field
                            of the event in dld mode */
  unsigned char sum_length; /**< Length of time coordinate data field
                            of the event in dld mode. */
  unsigned char sign_counter_offset;
  unsigned char sign_counter_length;
  unsigned char reserved[14]; /**< Reserved fields. Must not be used. */
  unsigned char flow_control_flags; /**< Flow control flag data field. */
};

/**
 * @brief Pipe type.
 */
enum sc_pipe_type_t
{
  TDC_HISTO = 0, /**< Used to get pipe with tdc histo data */
  DLD_IMAGE_XY = 1, /**< Used to get 2d image data */
  DLD_IMAGE_XT = 2, /**< Used to get x-t image data */
  DLD_IMAGE_YT = 3, /**< Used to get y-t image data */
  DLD_IMAGE_3D = 4, /**< Used to get 3d (x,y,t cube) image data */
  DLD_SUM_HISTO = 5, /**< Used to get dld time histogram data */
  STATISTICS = 6, /**< Used to get statistics for last exposure */
  TMSTAMP_TDC_HISTO = 7,
  TDC_STATISTICS = 8,
  DLD_STATISTICS = 9,
  USER_CALLBACKS = 10, /**< Used to receive lists of events (TDCs, DLDs, and 
                            cameras) */
  DLD_IMAGE_XY_EXT = 11, /**< Used to get 2d image data with extended parameter 
                              set */
  BUFFERED_DATA_CALLBACKS = 12, /**< Used by Python SDK, to receive lists of  
                                   events, reduces the frequency of callbacks */
  PIPE_CAM_FRAMES = 13, /**< Used to receive camera frame meta data, 
                             and raw image frame data, use nullptr as 3rd arg
                             in sc_pipe_open2() */
  PIPE_CAM_BLOBS = 14, /**< Used to receive camera blob data, use nullptr as 3rd
                            arg in sc_pipe_open2() */
  USED_MEM_CALLBACKS = 100 /**< Used to monitor hardware memory usage level */
};

/**
 * @brief Parameters for DLD_IMAGE_XY pipe type.
 *
 * The size of one histogram in memory is roi.x * roi.y * element_size bytes,
 * where element_size depends on the choice of depth.
 */
struct sc_pipe_dld_image_xy_params_t
{
  enum bitsize_t depth; /**< Data type of histogram elements. */
  int channel; /**< Filter by channel if >= 0 (-1 for normal detector) */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  struct sc3du_t binning; /**< x,y,time are divided by the respective binnings
                               before adding into the histogram. */
  struct roi_t roi; /**< Region of interest from which image will be built. */
  unsigned int accumulation_ms; /**< Accumulation time. */
  void *allocator_owner; /**< A pointer chosen by the user that gets passed back
                              into the allocator_cb */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for DLD_IMAGE_XT pipe type.
 *
 * The size of one histogram in memory is roi.x * roi.t * element_size bytes,
 * where element_size depends on the choice of depth.
 */
struct sc_pipe_dld_image_xt_params_t
{
  enum bitsize_t depth; /**< Data type of histogram elements. */
  int channel; /**< Filter by channel if >= 0 (-1 for normal detector) */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  struct sc3du_t binning; /**< x,y,time are divided by the respective binnings
                               before adding into the histogram. */
  struct roi_t roi; /**< Region of interest from which image will be built. */
  unsigned int accumulation_ms;  /**< Accumulation time. */
  void *allocator_owner; /**< A pointer chosen by the user that gets passed back
                              into the allocator_cb */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for DLD_IMAGE_YT pipe type.
 *
 * The size of one histogram in memory is roi.y * roi.t * element_size bytes,
 * where element_size depends on the choice of depth.
 */
struct sc_pipe_dld_image_yt_params_t
{
  enum bitsize_t depth; /**< Data type of histogram elements. */
  int channel; /**< Filter by channel if >= 0 (-1 for normal detector) */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  struct sc3du_t binning;  /**< x,y,time are divided by the respective binnings
                               before adding into the histogram. */
  struct roi_t roi; /**< Region of interest from which image will be built. */
  unsigned int accumulation_ms;  /**< Accumulation time. */
  void *allocator_owner; /**< A pointer chosen by the user that gets passed back
                              into the allocator_cb */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for DLD_IMAGE_3D pipe type.
 *
 * The size of one histogram in memory is roi.x * roi.y * roi.t * element_size,
 * where element_size depends on the choice of depth.
 * Please note, that modulo, binning and roi settings are applied in the same
 * order like set in the structure. e.g. first modulo is applied,
 * then binning and then roi.
 */
struct sc_pipe_dld_image_3d_params_t
{
  enum bitsize_t depth; /**< Data type of histogram elements. */
  int channel; /**< Filter by channel if >= 0 (-1 for normal detector) */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  struct sc3du_t binning;  /**< x,y,time are divided by the respective binnings
                               before adding into the histogram. */
  struct roi_t roi; /**< Region of interest from which image will be built. */
  unsigned int accumulation_ms;  /**< Accumulation time. */
  void *allocator_owner; /**< A pointer chosen by the user that gets passed back
                              into the allocator_cb */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for DLD_SUM_HISTO pipe type -- a 1D histogram with a time
 * axis integrating over a region of interest with respect to detector
 * positions. The size of one histogram in memory is roi.t * depth bytes.
 */
struct sc_pipe_dld_sum_histo_params_t
{
  enum bitsize_t depth; /**< Data type of histogram elements. */
  int channel; /**< Filter by channel if >= 0 (-1 for normal detector). */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  struct sc3du_t binning;  /**< x,y,time are divided by the respective binnings
                               before adding into the histogram. */
  struct roi_t roi; /**< Region of interest from which histogram will be built. */
  unsigned int accumulation_ms; /**< Accumulation time. */
  void *allocator_owner; /**< A pointer chosen by the user that gets passed back
                              into the allocator_cb */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for TDC_HISTO pipe type.
 *
 * The size of one histogram in memory is size * depth bytes.
 */
struct sc_pipe_tdc_histo_params_t
{
  enum bitsize_t depth; /**< Bits per histogram element in memory. */
  unsigned int channel; /**< Channel is used to build histogram. */
  unsigned long long modulo; /**< if > 0, apply modulo operation to time value
                                  before inserting it into the histogram. */
  unsigned int binning; /**< Histogram time binning. */
  unsigned long long offset; /**< Histogram start offset in time bins (see sc_tdc_get_binsize2()). */
  unsigned int size; /**< Histogram size in time bins (see above). */
  unsigned int accumulation_ms; /**< Accumulation time. */
  void *allocator_owner; /** Parameter for the allocator_cb function. */
  /**
   * User-provided allocator function, that the library calls to allocate memory
   * for data. If set to NULL, the library uses internal memory allocation.
   * The allocator_owner pointer is passed into the allocator_cb as the first
   * argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

/**
 * @brief Parameters for STATISTICS pipe type.
 *
 * This pipes delivers data of the type struct statistics_t. The memory
 * requirement for each such struct is sizeof(@ref statistics_t) = 1024 bytes.
 */
struct sc_pipe_statistics_params_t
{
  void *allocator_owner;
  /**
   * Used to allocate memory for data. If NULL - direct memory allocation.
   * allocator_owner field will used as first argument during the call.
   */
  int (*allocator_cb)(void *, void **);
};

struct sc_pipe_tdc_stat_params_t
{
  void *allocator_owner;
  int (*allocator_cb) (void *, void **);
  int channel_number;
};

struct sc_pipe_dld_stat_params_t
{
  void *allocator_owner;
  int (*allocator_cb) (void *, void **);
  int device_number;
};

enum sc_pipe_param_extension_type
{
  SC_PIPE_PARAM_EXTENSION_TYPE_IMAGE_SOURCE
};

struct sc_pipe_image_source
{
  enum sc_pipe_param_extension_type type;
  void *extension;
  enum sc_ImageSource {
    EVENTS,
    RAMDATA,
    BOTH
  } source;
};

struct sc_pipe_dld_image_xy_ext_params_t
{
  struct sc_pipe_dld_image_xy_params_t base;
  void *extension;
};

/**
  @brief Tdc data received from device.
*/
struct sc_TdcEvent
{
  unsigned subdevice; /**< Subdevice where TDC event occurred.*/
  unsigned channel; /**< Tdc channel where TDC event occurred.*/
  unsigned long long start_counter; /**< Start pulse counter.*/
  unsigned long long time_tag; /**< The 'tag' value related to the 'Tag In' 
                                    hardware input.*/
  unsigned long long time_data; /**< Time of the TDC event in multiple of time 
                                     bins referred to the last start pulse.*/
  unsigned long long sign_counter; /**< counts up when a pulse is applied to the
                                        respective hardware input */
};

/**
  @brief Dld data received from device.
*/
struct sc_DldEvent
{
  unsigned long long start_counter; /**< Start pulse counter.*/
  unsigned long long time_tag; /**< The 'tag' value related to the 'Tag In' 
                                    hardware input .*/
  unsigned subdevice;     /**< Subdevice where DLD event occurred.*/
  unsigned channel;       /**< Often unused, enumerates segments in multi-segment 
                               detectors */
  unsigned long long sum; /**< Time of the DLD event in multiple of time bins 
                               referred to the last start pulse.*/
  unsigned short dif1;    /**< X coordinate of dld event.*/
  unsigned short dif2;    /**< Y coordinate of dld event.*/
  unsigned master_rst_counter; /**< counts up when a pulse is applied to the 
                                    respective hardware input */
  unsigned short adc;     /**< the digitized 16-bit value of the ADC input */
  unsigned short signal1bit; /**< either 0 or 1 depending on the voltage level 
                                  applied to the State input */
};

//----- struct sc_pipe_callbacks -----------------------------------------------
/**
  @brief Set of callback functions provided by the application developer to be 
  called from the scTDC library for a USER_CALLBACKS pipe. The callback 
  functions handle various events and the reception of TDC or DLD data. 
  They are called synchronously from the library thread that decodes the
  protocol stream from the hardware, thus preserving correct chronological 
  order of data and events.

  Any callback pointers may be set to zero if the application does not need 
  information about one or another event or data. For example, the tdc_event 
  callback can be set to zero for DLD applications. (version 1.3017.5 fixes
  crashes for the case where 'start_of_measure' is zero).
  
  Callback functions must limit the amount of time they need to execute, since
  the internal decoding of the hardware protocol stream pauses until the 
  callback function returns. For best performance, you may decide to put the
  data you are interested in into a memory buffer that can be processed by a 
  different thread, afterwards.
  
  Furthermore, it is generally not possible to start new measurements from 
  within the 'end_of_measure' callback. To work around this limit, it is 
  recommended to use some notification mechanism into your main thread to 
  schedule the start of the next measurement.
  
  Make sure, not to pass this struct directly into the sc_pipe_open2() function.
  The correct usage is to take the address of a sc_pipe_callbacks variable, set
  this address in the 'callbacks' field of a sc_pipe_callback_params_t variable 
  and pass the address of this latter variable to sc_pipe_open2().

  @see struct sc_DeviceProperties3, struct sc_pipe_callback_params_t
*/
struct sc_pipe_callbacks
{
  /** Private user pointer that is passed back into the callback functions */
  void *priv;
  /** Called when the start of a measurement appears in the hardware protocol stream. */
  void (*start_of_measure) (void *priv);
  /** Called when the end of a measurement appears in the hardware protocol stream. */
  void (*end_of_measure) (void *priv);
  /** Called at points where the hardware recorded an ellapsed millisecond.*/
  void (*millisecond_countup) (void *priv);
  /** Called when statistics info appears in the hardware protocol stream, usually at the end of measurements.*/
  void (*statistics) (void *priv, const struct statistics_t *stat);
  /** Called for transmission of TDC events (event_array_len = number of events) */
  void (*tdc_event)
    (void *priv, const struct sc_TdcEvent *const event_array, size_t event_array_len);
  /** Called for transmission of DLD events (event_array_len = number of events) */
  void (*dld_event)
    (void *priv, const struct sc_DldEvent *const event_array, size_t event_array_len);
};


/**
  @brief Parameters for USER_CALLBACKS pipe type, which is to be passed by 
  pointer into the sc_pipe_open2().
 */
struct sc_pipe_callback_params_t
{
  struct sc_pipe_callbacks *callbacks;
};

/**
  @brief Structure that is passed into a callback function for
  a BUFFERED_DATA_CALLBACKS pipe.

  Any of the pointer members may be null, indicating that no data is
  available.

  @see struct sc_pipe_buf_callbacks_params_t.
*/
struct sc_pipe_buf_callback_args
{
  /* any of the pointer members may be null, indicating
   * that no data is available */
  unsigned long long event_index;    /**< Index of the first event.*/
  unsigned long long* som_indices;   /**< Start of measurement indices.*/
  unsigned long long* ms_indices;    /**< Millisecond indices.*/
  unsigned* subdevice;               /**< Subdevice values.*/
  unsigned* channel;                 /**< Channel values.*/
  unsigned long long* start_counter; /**< Start counter values.*/
  unsigned* time_tag;                /**< Time Tag values.*/
  unsigned* dif1;                    /**< Dif1 values / x detector coord.*/
  unsigned* dif2;                    /**< Dif2 values / y detector coord.*/
  unsigned long long* time;          /**< Time values.*/
  unsigned* master_rst_counter;      /**< Master reset counter values.*/
  int* adc;                          /**< ADC values.*/
  unsigned short* signal1bit;        /**< State input values.*/
  unsigned som_indices_len;          /**< length of som_indices array.*/
  unsigned ms_indices_len;           /**< length of ms_indices array.*/
  unsigned data_len;                 /**< length of each data array.*/
  unsigned char reserved[12];        /**< future use.*/
};

enum sc_data_field_t
{
  SC_DATA_FIELD_SUBDEVICE          = 0x0001u,
  SC_DATA_FIELD_CHANNEL            = 0x0002u,
  SC_DATA_FIELD_START_COUNTER      = 0x0004u,
  SC_DATA_FIELD_TIME_TAG           = 0x0008u,
  SC_DATA_FIELD_DIF1               = 0x0010u,
  SC_DATA_FIELD_DIF2               = 0x0020u,
  SC_DATA_FIELD_TIME               = 0x0040u,
  SC_DATA_FIELD_MASTER_RST_COUNTER = 0x0080u,
  SC_DATA_FIELD_ADC                = 0x0100u,
  SC_DATA_FIELD_SIGNAL1BIT         = 0x0200u,
};

/**
  @brief Parameter structure to be passed as the third argument in 
  sc_pipe_open2 when creating a BUFFERED_DATA_CALLBACKS pipe.
   
  This kind of pipe may be useful for language bindings where invocation
  of callback functions is slow (e.g. Python). The pipe issues callbacks only
  once a configurable minimum number of events has been buffered -- so as to
  reduce the frequency of callbacks. Data is buffered in separate arrays of 
  basic datatypes and the event data fields to be buffered can be selected.

  @see enum sc_data_field_t, BUFFERED_DATA_CALLBACKS
*/
struct sc_pipe_buf_callbacks_params_t
{
  /** Private data.*/
  void *priv;
  /** Callback providing buffered data.*/
  void (*data)(void *priv, const sc_pipe_buf_callback_args* const);
  /** Callback signalizing end of measurement. If callback returns true,
   *  all currently buffered data will be immediately transferred via an
   *  invocation of the 'data' callback.*/
  bool (*end_of_measurement)(void *priv);
  /** select which of the event data fields to buffer.*/
  unsigned data_field_selection;
  /** maximum number of entries per data field to be buffered.*/
  unsigned max_buffered_data_len;
  /** if 0, buffer TDC events; if 1, buffer DLD events */
  int dld_events;
  /** version must be set to zero (enables future extensions) */
  int version;
  /** future use.*/
  unsigned char reserved[24];
};

struct sc_pipe_used_mem_callbacks_params_t
{
  /** private data */
  void *priv;
  void (*used_mem)(void *priv, const unsigned used_mem_kb_value);
};

/**
 * @var sc_mask64
 * @brief Used to help user to extract data fields from the event when using
 * the sc_tdc_pipe_XYZ() functions.
 * @details Used in case of 64 bit event length.
 * @deprecated use sc_pipe_open2() with USER_CALLBACKS from enum sc_pipe_type_t
 * (defined in scTDC_types.h) as a replacement to the sc_tdc_pipe_XYZ() 
 * functions.
 */
const unsigned long long sc_mask64[] = {
  0x0000000000000000ULL,
  0x0000000000000001ULL,
  0x0000000000000003ULL,
  0x0000000000000007ULL,
  0x000000000000000FULL,
  0x000000000000001FULL,
  0x000000000000003FULL,
  0x000000000000007FULL,
  0x00000000000000FFULL,
  0x00000000000001FFULL,
  0x00000000000003FFULL,
  0x00000000000007FFULL,
  0x0000000000000FFFULL,
  0x0000000000001FFFULL,
  0x0000000000003FFFULL,
  0x0000000000007FFFULL,
  0x000000000000FFFFULL,
  0x000000000001FFFFULL,
  0x000000000003FFFFULL,
  0x000000000007FFFFULL,
  0x00000000000FFFFFULL,
  0x00000000001FFFFFULL,
  0x00000000003FFFFFULL,
  0x00000000007FFFFFULL,
  0x0000000000FFFFFFULL,
  0x0000000001FFFFFFULL,
  0x0000000003FFFFFFULL,
  0x0000000007FFFFFFULL,
  0x000000000FFFFFFFULL,
  0x000000001FFFFFFFULL,
  0x000000003FFFFFFFULL,
  0x000000007FFFFFFFULL,
  0x00000000FFFFFFFFULL,
  0x00000001FFFFFFFFULL,
  0x00000003FFFFFFFFULL,
  0x00000007FFFFFFFFULL,
  0x0000000FFFFFFFFFULL,
  0x0000001FFFFFFFFFULL,
  0x0000003FFFFFFFFFULL,
  0x0000007FFFFFFFFFULL,
  0x000000FFFFFFFFFFULL,
  0x000001FFFFFFFFFFULL,
  0x000003FFFFFFFFFFULL,
  0x000007FFFFFFFFFFULL,
  0x00000FFFFFFFFFFFULL,
  0x00001FFFFFFFFFFFULL,
  0x00003FFFFFFFFFFFULL,
  0x00007FFFFFFFFFFFULL,
  0x0000FFFFFFFFFFFFULL,
  0x0001FFFFFFFFFFFFULL,
  0x0003FFFFFFFFFFFFULL,
  0x0007FFFFFFFFFFFFULL,
  0x000FFFFFFFFFFFFFULL,
  0x001FFFFFFFFFFFFFULL,
  0x003FFFFFFFFFFFFFULL,
  0x007FFFFFFFFFFFFFULL,
  0x00FFFFFFFFFFFFFFULL,
  0x01FFFFFFFFFFFFFFULL,
  0x03FFFFFFFFFFFFFFULL,
  0x07FFFFFFFFFFFFFFULL,
  0x0FFFFFFFFFFFFFFFULL,
  0x1FFFFFFFFFFFFFFFULL,
  0x3FFFFFFFFFFFFFFFULL,
  0x7FFFFFFFFFFFFFFFULL,
  0xFFFFFFFFFFFFFFFFULL,
};

/**
 * @var sc_mask32
 * @brief Used to help user to extract data fields from the event when using
 * the sc_tdc_pipe_XYZ() functions.
 * @details Used in case of 32 bit event length
 * @deprecated use sc_pipe_open2() with USER_CALLBACKS from enum sc_pipe_type_t
 * (defined in scTDC_types.h) as a replacement to the sc_tdc_pipe_XYZ() 
 * functions.
 */
const unsigned int sc_mask32[] = {
  0x00000000,
  0x00000001,
  0x00000003,
  0x00000007,
  0x0000000F,
  0x0000001F,
  0x0000003F,
  0x0000007F,
  0x000000FF,
  0x000001FF,
  0x000003FF,
  0x000007FF,
  0x00000FFF,
  0x00001FFF,
  0x00003FFF,
  0x00007FFF,
  0x0000FFFF,
  0x0001FFFF,
  0x0003FFFF,
  0x0007FFFF,
  0x000FFFFF,
  0x001FFFFF,
  0x003FFFFF,
  0x007FFFFF,
  0x00FFFFFF,
  0x01FFFFFF,
  0x03FFFFFF,
  0x07FFFFFF,
  0x0FFFFFFF,
  0x1FFFFFFF,
  0x3FFFFFFF,
  0x7FFFFFFF,
  0xFFFFFFFF,
};


/**
 * @brief Used as argument in functions sc_tdc_is_event(), sc_tdc_is_event2().
 */
enum sc_event_type_index {
  SC_TDC_SIGN_START = 0, /**< Tdc event is start sign. */
  SC_TDC_SIGN_MILLISEC = 1, /**< Tdc event is millisecond sign. */
  SC_TDC_SIGN_STAT = 2 /**< Tdc event is beginning of statistics sign. */
};

/**
 * @brief Logging level.
 * @see sc_dbg_set_logger()
 * @deprecated Is not used anymore.
 */
enum sc_LoggerFacility {UNUSED}; //TODO: Remove

/**
 * @struct sc_Logger
 * @brief Logger descriptor used for debug.
 * @details The structure
 * @see sc_dbg_set_logger()
 */
struct sc_Logger
{
  void *private_data; /**< Private data of the external logger. */
  /**
   * @brief Logger callback function.
   * @param pd private_data field.
   * @param sender Sender of the debug message to be logger.
   * @param msg Message itself.
   */
  void (*do_log)(void *pd, const char *sender, const char *msg);
};

struct sc_PipeCbf
{
  void (*cb)(void *);
  void *private_data;
};

struct sc_ConfigLine
{
  const char *section;
  const char *key;
  const char *value;
};

/**
 * @brief Device Properties 1.
 *
 * If mentioned below parameters are not available they are being set to 0.
 * The x and y fields in detector_size report the size of the aperture, scaled
 * by StretchX/Y + HardwareBinningX/Y parameters from the ini file --- to the
 * effect, that opening a DLD_IMAGE_XY histogram with the sizes for x and y
 * set to detector_size.x and detector_size.y display the full image as cropped
 * by the aperture settings.
 * The pixel_size_x / pixel_size_y values are taken verbatim from the ini file
 * and it is the responsibility of whoever adapts StretchX/Y and
 * HardwareBinningX/Y to update these values, accordingly, such that they
 * yield the physical size corresponding to one pixel in a DLD_IMAGE_XY
 * histogram.
 */
struct sc_DeviceProperties1
{
  struct sc3du_t detector_size; /**< Physical detector size in pixels */

  double pixel_size_x; /**< Pixel size in x direction in mm */
  double pixel_size_y; /**< Pixel size in y direction in mm */
  double pixel_size_t; /**< Pixel size in time direction in ns */
};

/**
 * @brief Device Properties 2.
*/
struct sc_DeviceProperties2
{
  int tdc_channel_number; /**< Number of tdc channel device has */
};

/**
  @brief Device Properties 3.
 */
struct sc_DeviceProperties3
{
  size_t dld_event_size; /**< size of sc_DldEvent structure in bytes */
  size_t tdc_event_size; /**< size of sc_TdcEvent structure in bytes */
  size_t user_callback_size; /**< size of sc_pipe_callbacks structure in bytes */
};

/**
  @brief Device Properties 4.
 */
struct sc_DeviceProperties4
{
  unsigned auto_start_period; /**< automatically measured start period during
                                   initialization in units of time bins of the
                                   TDC / DLD */
  unsigned auto_modulo;       /**< automatically measured modulo during
                                   initialization */
};

/**
  @brief Device Properties 5.
 */
struct sc_DeviceProperties5
{
  unsigned tag_period; /**< averaged tag period (over ~1.6 seconds) when tag is
                            used as a timer, given in "tag units" (12.5 ns) */
};

/**
  @brief Device Properties 6
 */
struct sc_DeviceProperties6
{
  unsigned tag_max; /**< maximum of tag value (when used as timer) when it was
                         last reset to 0 by a pulse on "Tag In" */
};

/**
  @brief Cmos and Smoother parameters structure.
*/

struct sc_CmosSmootherParameters
{
  enum sc_ShutterMode {
    FULLY_EXTERNAL,  //!< start and stop of frames controlled by hardware input ("wire")
    IMMEDIATE_START_INTERNAL_TIMER, //!< fully software-controlled frames
    IMMEDIATE_START_INTERNAL_TIMER_MULTIPLE_SLOPES, //!< not in use anymore
    EXTERNAL_START_INTERNAL_TIMER,  //!< start of frames by wire, stop by software
    EXTERNAL_START_INTERNAL_TIMER_MULTIPLE_SLOPES,  //!< not in use anymore
    IMMEDIATE_START_EXTERNAL_FINISH //!< start of frames by software, stop by wire
  } shutter_mode; /**< controls triggering of camera frames by hardware input */
  unsigned int single_slope_us; /**< the exposure of a single camera frame */
  unsigned int dual_slope_us; /**< not in use anymore */
  unsigned int triple_slope_us; /**< not in use anymore */
  unsigned int frame_count; /**< sets the number of frames per measurement */
  unsigned int analog_gain; /**< the analog gain parameter of the sensor */
  double digital_gain;      /**< not in use anymore */
  unsigned int black_offset; /**< the black offset parameter of the sensor */
  int black_cal_offset;      /**< not in use anymore */
  unsigned int smoother_shift1; /**< intensity scale-down after application of
                                     smoother_pixel_mask1 */
  unsigned char smoother_pixel_mask1 [8][8]; /**< first smoother mask */
  unsigned int smoother_shift2; /**< intensity scale-down after application of
                                     smoother_pixel_mask2 */
  unsigned char smoother_pixel_mask2 [8][8]; /**< second smoother mask */
  unsigned char white_pixel_min; /**< threshold for white pixel suppression */
};

/**
  @brief Blob parameters structure.
*/

struct sc_BlobParameters
{
  unsigned int unbinning; //!< scales blob coordinates by a power of 2
  unsigned int dif_min_top;  /**< if >= 0, activate blob mode, threshold
                                  condition for blob recognition */
  unsigned int dif_min_bottom; /**< if >= 0, activate blob mode, threshold
                                    condition for blob recognition */
  double z_scale_factor; /**< for future use, set it to 1.0, for now */
};

struct sc_flimTriggersCounters {
  unsigned long long pixelTriggerCounter;
  unsigned long long lineTriggerCounter;
  unsigned long long frameTriggerCounter;
};
