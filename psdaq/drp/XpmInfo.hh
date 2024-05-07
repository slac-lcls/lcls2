#pragma once

#include "rapidjson/document.h"

namespace Drp {
  static inline nlohmann::json xpmInfo(unsigned paddr) {
    int xpm  = (paddr >> 16) & 0xFF;
    int port = (paddr >>  0) & 0xFF;
    nlohmann::json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info; 
  }
};
