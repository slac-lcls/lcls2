/**
 * @file
 * @brief xpmInfo(), which turns an XPM link address word into connection-info JSON.
 */
#pragma once

#include "rapidjson/document.h"

namespace Drp {
  /** Return JSON with xpm_id set to bits 23-16 of paddr and xpm_port set to bits 7-0. */
  static inline nlohmann::json xpmInfo(unsigned paddr) {
    int xpm  = (paddr >> 16) & 0xFF;
    int port = (paddr >>  0) & 0xFF;
    nlohmann::json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info; 
  }
};
