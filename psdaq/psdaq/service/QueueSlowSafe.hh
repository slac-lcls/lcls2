/**
 * @file
 * @brief Includes QueueSlowSafeVx.hh when VXWORKS is defined, otherwise QueueSlowSafeUx.hh (QueueSS).
 */
#ifdef VXWORKS
#include "QueueSlowSafeVx.hh"
#else
#include "QueueSlowSafeUx.hh"
#endif
